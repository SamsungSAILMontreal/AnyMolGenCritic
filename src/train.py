import os
import sys
import argparse
from numpy.lib.arraysetops import unique
from rdkit import Chem
import random
import torch
import copy
from torch.utils.data import DataLoader
import shutil

import torch.distributed as dist
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
from pytorch_lightning.utilities import rank_zero_only
import moses
from moses.utils import disable_rdkit_log, enable_rdkit_log
from data.target_data import Data as TargetData
from torch.distributions.bernoulli import Bernoulli

from model.generator import CondGenerator
from data.dataset import get_cond_datasets, DummyDataset, merge_datasets
from props.properties import penalized_logp, MAE_properties, best_rewards_gflownet
from util import compute_sequence_cross_entropy, compute_property_accuracy, compute_sequence_accuracy, canonicalize
from evaluate.MCD.evaluator import TaskModel, compute_molecular_metrics
from moses.metrics.metrics import compute_intermediate_statistics
from joblib import dump, load
import numpy as np

class CondGeneratorLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super(CondGeneratorLightningModule, self).__init__()
        hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_datasets()
        self.setup_model()

    def setup_datasets(self):
        raw_dir = f"../resource/data/{self.hparams.dataset_name}"
        self.train_dataset, self.val_dataset, self.test_dataset = get_cond_datasets(dataset_name=self.hparams.dataset_name,
            raw_dir=raw_dir, randomize_order=self.hparams.randomize_order, 
            MAX_LEN=self.hparams.max_len, scaling_type=self.hparams.scaling_type, 
            gflownet=self.hparams.gflownet, 
            n_properties=self.hparams.n_properties, start_min=not self.hparams.start_random)
        print(self.train_dataset.scaler_std_properties.column_transformer.mean_)
        print(self.train_dataset.scaler_std_properties.column_transformer.var_)

        # Add a fine-tuning dataset
        if self.hparams.finetune_dataset_name != '':
            raw_dir2 = f"../resource/data/{self.hparams.finetune_dataset_name}"
            train_dataset2, val_dataset2, test_dataset2 = get_cond_datasets(dataset_name=self.hparams.finetune_dataset_name,
                raw_dir=raw_dir2, randomize_order=self.hparams.randomize_order, 
                MAX_LEN=self.hparams.max_len, scaling_type=self.hparams.scaling_type, 
                gflownet=self.hparams.gflownet, 
                n_properties=self.hparams.n_properties, start_min=not self.hparams.start_random)

            # Combined dataset
            train_dataset_combined = merge_datasets(self.train_dataset, train_dataset2)
            val_dataset_combined = merge_datasets(self.val_dataset, val_dataset2, scaler_std_properties=train_dataset_combined.scaler_std_properties, scaler_properties=train_dataset_combined.scaler_properties)
            test_dataset_combined = merge_datasets(self.test_dataset, test_dataset2, scaler_std_properties=train_dataset_combined.scaler_std_properties, scaler_properties=train_dataset_combined.scaler_properties)
            print(train_dataset_combined.scaler_std_properties.column_transformer.mean_)
            print(train_dataset_combined.scaler_std_properties.column_transformer.var_)

            # Qe want the scaler for the original or the fine-tuning dataset
            if self.hparams.finetune_scaler_vocab:
                train_dataset_combined.update_scalers(test_dataset2)
                val_dataset_combined.update_scalers(val_dataset2)
                test_dataset_combined.update_scalers(test_dataset2)
            else:
                train_dataset_combined.update_scalers(self.train_dataset)
                val_dataset_combined.update_scalers(self.val_dataset)
                test_dataset_combined.update_scalers(self.test_dataset)

            # combine both datasets vocabs and ensure good scaler, but keep the molecules and properties from the original dataset
            self.train_dataset.update_vocabs_scalers(train_dataset_combined)
            self.val_dataset.update_vocabs_scalers(val_dataset_combined)
            self.test_dataset.update_vocabs_scalers(test_dataset_combined)
            print(self.train_dataset.scaler_std_properties.column_transformer.mean_)
            print(self.train_dataset.scaler_std_properties.column_transformer.var_)

        print(f"--Atom vocabulary size--: {len(self.train_dataset.vocab.ATOM_TOKENS)}")
        print(self.train_dataset.vocab.ATOM_TOKENS)

        self.train_smiles_set = set(self.train_dataset.smiles_list)
        self.hparams.vocab = self.train_dataset.vocab

        def collate_fn(data_list):
            batched_mol_data, batched_cond_data = zip(*data_list)
            return TargetData.collate(batched_mol_data), torch.stack(batched_cond_data, dim=0)    

        self.collate_fn = collate_fn

    def setup_model(self):
        self.model = CondGenerator(
            num_layers=self.hparams.num_layers,
            emb_size=self.hparams.emb_size,
            nhead=self.hparams.nhead,
            dim_feedforward=self.hparams.dim_feedforward,
            input_dropout=self.hparams.input_dropout,
            dropout=self.hparams.dropout,
            disable_treeloc=self.hparams.disable_treeloc,
            disable_graphmask=self.hparams.disable_graphmask, 
            disable_valencemask=self.hparams.disable_valencemask,
            disable_counting_ring=self.hparams.disable_counting_ring,
            disable_random_prop_mask=self.hparams.disable_random_prop_mask,
            enable_absloc=self.hparams.enable_absloc,
            lambda_predict_prop=self.hparams.lambda_predict_prop,
            MAX_LEN=self.hparams.max_len,
            gpt=self.hparams.gpt,
            bias=not self.hparams.no_bias,
            rotary=self.hparams.rotary,
            rmsnorm=self.hparams.rmsnorm,
            swiglu=self.hparams.swiglu,
            expand_scale=self.hparams.expand_scale,
            special_init=self.hparams.special_init,
            n_properties=self.hparams.n_properties,
            vocab=self.hparams.vocab,
            n_correct=self.hparams.n_correct,
            cond_lin=self.hparams.cond_lin,
            cat_var_index=self.hparams.cat_var_index,
            cont_var_index=self.hparams.cont_var_index,
        )

    ### Dataloaders and optimizers
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=0 if self.hparams.test else self.hparams.num_workers,
            drop_last=False,
            persistent_workers=not self.hparams.test and self.hparams.num_workers > 0, pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0 if self.hparams.test else self.hparams.num_workers,
            drop_last=False,
            persistent_workers=not self.hparams.test and self.hparams.num_workers > 0, pin_memory=True,
        )

    def test_dataloader(self):
        self.test_step_pred_acc = []
        if self.hparams.no_test_step:
            return DataLoader(
                DummyDataset(),
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )
        elif self.hparams.test_on_train_data:
            dset = self.train_dataset
        else:
            dset = self.test_dataset
        return DataLoader(
            dset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers if self.hparams.test else 0,
            drop_last=False,
            persistent_workers=False, pin_memory=True,
        )

    def configure_optimizers(self):
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay
            )
        # Warmup + Cosine scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=self.hparams.warmup_steps,
        )
        n_iters_per_epoch = len(self.train_dataset) // self.hparams.batch_size # because drop_last=True we can round down
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_iters_per_epoch*self.trainer.fit_loop.max_epochs - self.hparams.warmup_steps,
            eta_min=self.hparams.lr*self.hparams.lr_decay
        )
        combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.hparams.warmup_steps])
        lr_scheduler = {
            "scheduler": combined_scheduler,
            'interval': 'step'
        }
        return [optimizer], [lr_scheduler]

    def shared_step(self, batched_data, whole_data_metrics=False):
        loss, statistics = 0.0, dict()

        # decoding
        batched_mol_data, batched_cond_data = batched_data
        logits, pred_prop = self.model(batched_mol_data, batched_cond_data)
        loss, loss_prop_cont, loss_prop_cat = compute_sequence_cross_entropy(logits, batched_mol_data[0], ignore_index=0, 
            prop=batched_cond_data, pred_prop=pred_prop, lambda_predict_prop=self.hparams.lambda_predict_prop, 
            cont_var_index=self.hparams.cont_var_index, cat_var_index=self.hparams.cat_var_index)

        statistics["loss/total"] = loss
        statistics["loss/class"] = loss - loss_prop_cont - loss_prop_cat
        statistics["loss/prop_cont"] = loss_prop_cont
        statistics["loss/prop_cat"] = loss_prop_cat
        statistics["acc/total"] = compute_sequence_accuracy(logits, batched_mol_data[0], ignore_index=0)[0]

        if self.hparams.lambda_predict_prop > 0:
            if whole_data_metrics: # computer over whole dataset
                b = batched_mol_data[0].shape[0]
                test_step_pred_acc = compute_property_accuracy(batched_mol_data[0], 
                    prop=batched_cond_data, pred_prop=pred_prop, 
                    cont_var_index=self.hparams.cont_var_index, cat_var_index=self.hparams.cat_var_index, mean=False)
                self.test_step_pred_acc.append(test_step_pred_acc)
            else: # computer over the mini-batch
                validation_step_pred_acc = compute_property_accuracy(batched_mol_data[0], 
                    prop=batched_cond_data, pred_prop=pred_prop, 
                    cont_var_index=self.hparams.cont_var_index, cat_var_index=self.hparams.cat_var_index, mean=True)
                for i, prop_acc in enumerate(validation_step_pred_acc):
                    statistics[f"acc/prop{i}"] = prop_acc
        return loss, statistics

    def training_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data, whole_data_metrics=False)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True, sync_dist=self.hparams.n_gpu > 1)
        return loss

    def validation_step(self, batched_data, batch_idx):
        if self.hparams.no_test_step:
            loss = 0.0
        else:
            loss, statistics = self.shared_step(batched_data, whole_data_metrics=False)
            for key, val in statistics.items():
                self.log(f"validation/{key}", val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)
        return loss

    def test_step(self, batched_data, batch_idx):
        if self.hparams.no_test_step:
            loss = 0.0
        else:
            loss, statistics = self.shared_step(batched_data, whole_data_metrics=True)
            if self.hparams.lambda_predict_prop > 0:
                test_step_pred_acc = torch.cat(self.test_step_pred_acc, dim=0) # [N, p_variables]
                if self.hparams.n_gpu > 1:
                    all_test_step_pred_acc = self.all_gather(test_step_pred_acc).view(-1, self.hparams.n_properties)
                else:
                    all_test_step_pred_acc = test_step_pred_acc
                all_test_step_pred_acc = all_test_step_pred_acc.mean(dim=0) # [p_variables]
                self.test_step_pred_acc.clear()
                for i, prop_acc in enumerate(all_test_step_pred_acc):
                    statistics[f"acc/prop{i}"] = prop_acc
            for key, val in statistics.items():
                self.log(f"validation/{key}", val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)
        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking and not self.hparams.test and self.current_epoch > 0: # can lead to valence errors when untrained due to bad choices
            if (self.current_epoch + 1) % self.hparams.check_sample_every_n_epoch == 0:
                if self.hparams.gflownet:
                    self.check_samples_gflownet()
                elif self.hparams.graph_dit_benchmark:
                    self.check_samples_dit() # cond on train properties with special metrics
                else:
                    if not self.hparams.only_ood: # We have different properties, do not use ranking
                        self.check_samples_uncond() # uncond
                    if not self.hparams.no_ood: 
                        self.check_samples_ood() # cond on ood values
                    else:
                        self.check_samples() # cond on train properties

    def on_test_epoch_end(self):
        if not self.trainer.sanity_checking and self.current_epoch > 0: # can lead to valence errors when untrained due to bad choices
            if self.hparams.gflownet:
                self.check_samples_gflownet()
            elif self.hparams.graph_dit_benchmark:
                self.check_samples_dit() # cond on train properties with special metrics
            else:
                if not self.hparams.only_ood: # We have different properties, do not use ranking
                    self.check_samples_uncond() # uncond
                if not self.hparams.no_ood: 
                    self.check_samples_ood() # cond on ood values
                else:
                    self.check_samples() # cond on train properties

    # Sample extreme values for each properties and evaluate the OOD generated molecules 
    def check_samples_ood(self):
        assert self.hparams.num_samples_ood % self.hparams.n_gpu == 0
        num_samples = self.hparams.num_samples_ood // self.hparams.n_gpu if not self.trainer.sanity_checking else 2
        assert len(self.hparams.ood_values) == 1 or len(self.hparams.ood_values) == 2*self.train_dataset.n_properties
        i_ = 0
        
        for idx in range(self.train_dataset.n_properties):
            for j in range(2): #
                if j == 0: # u + std
                    stats_name = f"sample_ood_{idx}_mean_plus_std"
                    if len(self.hparams.ood_values) <= 1:
                        properties_np = self.train_dataset.get_mean_plus_std_property(idx, std=4)
                        print(f'raw_prop: {self.train_dataset.scaler_properties.inverse_transform(properties_np)[0]}')
                    else:
                        properties_np = np.zeros((1, self.train_dataset.n_properties))
                        properties_np[:, idx] = self.hparams.ood_values[i_]
                        print(f'raw_prop: {properties_np[0]}')
                        if self.train_dataset.scaler_properties is not None:
                            properties_np = self.train_dataset.scaler_properties.transform(properties_np) # raw to whatever
                        i_ += 1
                else: # u - std
                    stats_name = f"sample_ood_{idx}_mean_minus_std"
                    if len(self.hparams.ood_values) <= 1:
                        properties_np = self.train_dataset.get_mean_plus_std_property(idx, std=-4)
                        print(f'raw_prop: {self.train_dataset.scaler_properties.inverse_transform(properties_np)[0]}')
                    else:
                        properties_np = np.zeros((1, self.train_dataset.n_properties))
                        properties_np[:, idx] = self.hparams.ood_values[i_]
                        print(f'raw_prop: {properties_np[0]}')
                        if self.train_dataset.scaler_properties is not None:
                            properties_np = self.train_dataset.scaler_properties.transform(properties_np) # raw to whatever
                        i_ += 1
                print(f'std_prop: {properties_np[0]}')
                print(stats_name)
                properties_np = np.repeat(properties_np, num_samples, axis=0)
                local_properties = torch.tensor(properties_np).to(device=self.device, dtype=torch.float32)
                mask_cond = [i != idx for i in range(self.train_dataset.n_properties)] # we only give the single property we care about to the model; the model can figure out the rest
                local_smiles_list, results, _ = self.sample_cond(local_properties, num_samples, temperature=self.hparams.temperature_ood, 
                    guidance=self.hparams.guidance_ood, guidance_rand=self.hparams.guidance_rand, mask_cond=mask_cond)
                #print('smiles')
                #print(local_smiles_list[0:5])

                # Gather results
                if self.hparams.n_gpu > 1:
                    global_smiles_list = [None for _ in range(self.hparams.n_gpu)]
                    dist.all_gather_object(global_smiles_list, local_smiles_list)
                    smiles_list = []
                    for i in range(self.hparams.n_gpu):
                        smiles_list += global_smiles_list[i]

                    global_properties = [torch.zeros_like(local_properties) for _ in range(self.hparams.n_gpu)]
                    dist.all_gather(global_properties, local_properties)
                    properties = torch.cat(global_properties, dim=0)
                else:
                    smiles_list = local_smiles_list
                    properties = local_properties

                idx_valid = []
                valid_smiles_list = []
                valid_mols_list = []
                for smiles in smiles_list:
                    if smiles is not None:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            idx_valid += [True]
                            valid_smiles_list += [smiles]
                            valid_mols_list += [mol]
                        else:
                            idx_valid += [False]
                    else:
                        idx_valid += [False]
                unique_smiles_set = set(valid_smiles_list)
                novel_smiles_list = [smiles for smiles in valid_smiles_list if smiles not in self.train_smiles_set]
                efficient_smiles_list = [smiles for smiles in unique_smiles_set if smiles in novel_smiles_list]
                statistics = dict()

                statistics[f"{stats_name}/valid"] = float(len(valid_smiles_list)) / self.hparams.num_samples_ood
                statistics[f"{stats_name}/unique"] = float(len(unique_smiles_set)) / len(valid_smiles_list)
                statistics[f"{stats_name}/novel"] = float(len(novel_smiles_list)) / len(valid_smiles_list)
                statistics[f"{stats_name}/efficient"] = float(len(efficient_smiles_list)) / self.hparams.num_samples_ood
                if self.train_dataset.scaler_properties is not None:
                    properties_unscaled = torch.tensor(self.train_dataset.scaler_properties.inverse_transform(properties[idx_valid].cpu().numpy())).to(device=self.device, dtype=properties.dtype)
                else:
                    properties_unscaled = properties[idx_valid]
                print(properties_unscaled[0])
                statistics[f"{stats_name}/Min_MAE"], statistics[f"{stats_name}/Min10_MAE"], statistics[f"{stats_name}/Min100_MAE"] = MAE_properties(valid_mols_list, properties=properties_unscaled[:, idx].unsqueeze(1), properties_idx=[idx]) # molwt, LogP, QED
                print(statistics[f"{stats_name}/valid"])
                print(statistics[f"{stats_name}/Min_MAE"])
                print(statistics[f"{stats_name}/Min10_MAE"])
                print(statistics[f"{stats_name}/Min100_MAE"])
                for key, val in statistics.items():
                    self.log(key, val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)
            
    # Sample conditional on a property
    def sample_cond(self, properties, num_samples, guidance, guidance_rand, temperature, mask_cond=None):
        print("Sample_cond")
        offset = 0
        results = []
        loss_prop = []
        self.model.eval()
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            batched_cond_data = properties[offset:(offset+cur_num_samples), :]
            offset += cur_num_samples
            print(offset)
            data_list, loss_prop_ = self.model.decode(batched_cond_data, max_len=self.hparams.max_len, device=self.device, mask_cond=mask_cond,
                temperature=temperature, guidance=guidance, guidance_rand=guidance_rand,
                top_k=self.hparams.top_k, best_out_of_k=self.hparams.best_out_of_k,
                predict_prop=self.hparams.lambda_predict_prop > 0, return_loss_prop=self.hparams.lambda_predict_prop > 0,
                allow_empty_bond=not self.hparams.not_allow_empty_bond,)
            if loss_prop_ is not None:
                loss_prop += [loss_prop_]
            results.extend((data.to_smiles(), "".join(data.tokens), data.error) for data in data_list)
        if loss_prop_ is not None:
            loss_prop = torch.cat(loss_prop, dim=0)
        self.model.train()
        disable_rdkit_log()
        smiles_list = [canonicalize(elem[0]) for elem in results]
        enable_rdkit_log()

        return smiles_list, results, loss_prop

    # Generate molecules conditional on random properties from the test dataset
    def sample(self, num_samples):
        print("Sample")
        my_loader = DataLoader(
            self.test_dataset,
            batch_size=self.hparams.sample_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers-1,
            drop_last=False,
            persistent_workers=False, pin_memory=True,
        )
        train_loader_iter = iter(my_loader)

        offset = 0
        results = []
        loss_prop = []
        properties = None
        self.model.eval()
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples
            print(offset)
            try:
                _, batched_cond_data = next(train_loader_iter)
            except:
                train_loader_iter = iter(my_loader)
                _, batched_cond_data = next(train_loader_iter)
            while batched_cond_data.shape[0] < cur_num_samples:
                try:
                    _, batched_cond_data_ = next(train_loader_iter)
                except:
                    train_loader_iter = iter(my_loader)
                    _, batched_cond_data_ = next(train_loader_iter)
                batched_cond_data = torch.cat((batched_cond_data, batched_cond_data_), dim=0)
            batched_cond_data = batched_cond_data[:cur_num_samples,:].to(device=self.device)
            if properties is None:
                properties = batched_cond_data
            else:
                properties = torch.cat((properties, batched_cond_data), dim=0)
            data_list, loss_prop_ = self.model.decode(batched_cond_data, max_len=self.hparams.max_len, device=self.device, 
                temperature=self.hparams.temperature, guidance=self.hparams.guidance, guidance_rand=self.hparams.guidance_rand,
                top_k=self.hparams.top_k, best_out_of_k=self.hparams.best_out_of_k,
                predict_prop=self.hparams.lambda_predict_prop > 0,
                allow_empty_bond=not self.hparams.not_allow_empty_bond)
            if loss_prop_ is not None:
                loss_prop += [loss_prop_]
            results.extend((data.to_smiles(), "".join(data.tokens), data.error) for data in data_list)
        if loss_prop_ is not None:
            loss_prop = torch.cat(loss_prop, dim=0)
        self.model.train()
        disable_rdkit_log()
        smiles_list = [canonicalize(elem[0]) for elem in results]
        enable_rdkit_log()

        return smiles_list, results, properties, loss_prop

    def check_samples(self):
        assert self.hparams.num_samples % self.hparams.n_gpu == 0
        num_samples = self.hparams.num_samples // self.hparams.n_gpu if not self.trainer.sanity_checking else 2
        local_smiles_list, results, local_properties, _ = self.sample(num_samples)

        if self.hparams.n_gpu > 1:
            # Gather results
            global_smiles_list = [None for _ in range(self.hparams.n_gpu)]
            dist.all_gather_object(global_smiles_list, local_smiles_list)
            smiles_list = []
            for i in range(self.hparams.n_gpu):
                print(i)
                print(len(global_smiles_list[i]))
                smiles_list += global_smiles_list[i]

            global_properties = [torch.zeros_like(local_properties) for _ in range(self.hparams.n_gpu)]
            dist.all_gather(global_properties, local_properties)
            properties = torch.cat(global_properties, dim=0)
        else:
            smiles_list = local_smiles_list
            properties = local_properties

        print('metrics')
        #
        idx_valid = []
        valid_smiles_list = []
        valid_mols_list = []
        for smiles in smiles_list:
            if smiles is not None:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    idx_valid += [True]
                    valid_smiles_list += [smiles]
                    valid_mols_list += [mol]
                else:
                    idx_valid += [False]
            else:
                idx_valid += [False]
                
        unique_smiles_set = set(valid_smiles_list)
        novel_smiles_list = [smiles for smiles in valid_smiles_list if smiles not in self.train_smiles_set]
        efficient_smiles_list = [smiles for smiles in unique_smiles_set if smiles in novel_smiles_list]
        statistics = dict()
        statistics["sample/valid"] = float(len(valid_smiles_list)) / self.hparams.num_samples
        statistics["sample/unique"] = float(len(unique_smiles_set)) / len(valid_smiles_list)
        statistics["sample/novel"] = float(len(novel_smiles_list)) / len(valid_smiles_list)
        statistics["sample/efficient"] = float(len(efficient_smiles_list)) / self.hparams.num_samples
        if self.train_dataset.scaler_properties is not None:
            properties_unscaled = torch.tensor(self.train_dataset.scaler_properties.inverse_transform(properties[idx_valid].cpu().numpy())).to(dtype=torch.float32, device=self.device)
        else:
            properties_unscaled = properties[idx_valid]
        statistics["sample/Min_MAE"], statistics[f"sample/Min10_MAE"], statistics[f"sample/Min100_MAE"] = MAE_properties(valid_mols_list, properties=properties_unscaled) # molwt, LogP, QED
        print(statistics["sample/Min_MAE"])
        print(statistics["sample/Min10_MAE"])
        print(statistics["sample/Min100_MAE"])

        #
        for key, val in statistics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)
        
        if len(valid_smiles_list) > 0:
            torch.backends.cudnn.enabled = False
            print('get-all-metrics')
            moses_statistics = moses.get_all_metrics(
                smiles_list, 
                n_jobs=self.hparams.num_workers-1,
                device=str(self.device), 
                train=self.train_dataset.smiles_list, 
                test=self.test_dataset.smiles_list,
            )
            print('get-all-metrics done')
            for key in moses_statistics:
                self.log(f"sample/moses/{key}", moses_statistics[key], on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)#, rank_zero_only=True)
            torch.backends.cudnn.enabled = True

    # Generate molecules unconditional
    def sample_uncond(self, num_samples):
        print("Sample Uncond")
        offset = 0
        results = []
        loss_prop = []
        self.model.eval()
        batched_cond_data = torch.zeros(self.hparams.sample_batch_size, 3, device=self.device)
        mask_cond = [True for i in range(self.train_dataset.n_properties)] # mask everything
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples
            print(offset)
            data_list, loss_prop_ = self.model.decode(batched_cond_data[:cur_num_samples], mask_cond=mask_cond, max_len=self.hparams.max_len, device=self.device, 
                temperature=self.hparams.temperature, guidance=self.hparams.guidance, guidance_rand=self.hparams.guidance_rand,
                top_k=self.hparams.top_k, best_out_of_k=self.hparams.best_out_of_k,
                predict_prop=self.hparams.lambda_predict_prop > 0,
                allow_empty_bond=not self.hparams.not_allow_empty_bond)
            if loss_prop_ is not None:
                loss_prop += [loss_prop_]
            results.extend((data.to_smiles(), "".join(data.tokens), data.error) for data in data_list)
        if loss_prop_ is not None:
            loss_prop = torch.cat(loss_prop, dim=0)
        self.model.train()
        print("Sample Uncond -done-")
        disable_rdkit_log()
        smiles_list = [canonicalize(elem[0]) for elem in results]
        enable_rdkit_log()

        return smiles_list, results, loss_prop

    def check_samples_uncond(self):
        assert self.hparams.num_samples % self.hparams.n_gpu == 0
        num_samples = self.hparams.num_samples // self.hparams.n_gpu if not self.trainer.sanity_checking else 2
        local_smiles_list, results, _ = self.sample_uncond(num_samples)

        if self.hparams.n_gpu > 1:
            # Gather results
            global_smiles_list = [None for _ in range(self.hparams.n_gpu)]
            dist.all_gather_object(global_smiles_list, local_smiles_list)
            smiles_list = []
            for i in range(self.hparams.n_gpu):
                print(i)
                print(len(global_smiles_list[i]))
                smiles_list += global_smiles_list[i]
        else:
            smiles_list = local_smiles_list

        print('metrics')
        #
        idx_valid = [smiles is not None for smiles in smiles_list]
        valid_smiles_list = [smiles for i, smiles in enumerate(smiles_list) if idx_valid]
        unique_smiles_set = set(valid_smiles_list)
        novel_smiles_list = [smiles for smiles in valid_smiles_list if smiles not in self.train_smiles_set]
        efficient_smiles_list = [smiles for smiles in unique_smiles_set if smiles in novel_smiles_list]
        statistics = dict()
        statistics["sample_uncond/valid"] = float(len(valid_smiles_list)) / self.hparams.num_samples
        statistics["sample_uncond/unique"] = float(len(unique_smiles_set)) / len(valid_smiles_list)
        statistics["sample_uncond/novel"] = float(len(novel_smiles_list)) / len(valid_smiles_list)
        statistics["sample_uncond/efficient"] = float(len(efficient_smiles_list)) / self.hparams.num_samples

        #
        for key, val in statistics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)
        
        if len(valid_smiles_list) > 0:
            torch.backends.cudnn.enabled = False
            print('get-all-metrics uncond')
            moses_statistics = moses.get_all_metrics(
                smiles_list, 
                n_jobs=self.hparams.num_workers-1,
                device=str(self.device), 
                train=self.train_dataset.smiles_list, 
                test=self.test_dataset.smiles_list,
            )
            print('get-all-metrics uncond done')
            for key in moses_statistics:
                self.log(f"sample_uncond/moses/{key}", moses_statistics[key], on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)#, rank_zero_only=True)
            torch.backends.cudnn.enabled = True

    def check_samples_gflownet(self):
        assert self.hparams.num_samples_gflownet % self.hparams.n_gpu == 0
        num_samples = self.hparams.num_samples_gflownet // self.hparams.n_gpu if not self.trainer.sanity_checking else 2
        stats_name = f"sample"

        properties_np = np.array([self.hparams.gflownet_values]) # desired value to get reward zero
        print(properties_np)
        if self.train_dataset.scaler_properties is not None:
            properties_np = self.train_dataset.scaler_properties.transform(properties_np) # raw to whatever
            print(properties_np)
        properties_np = np.repeat(properties_np, num_samples, axis=0)
        local_properties = torch.tensor(properties_np).to(device=self.device, dtype=torch.float32)
        local_smiles_list, results, _ = self.sample_cond(local_properties, num_samples, 
            temperature=self.hparams.temperature, guidance=self.hparams.guidance, guidance_rand=self.hparams.guidance_rand, mask_cond=None)
        #print('smiles')
        #print(local_smiles_list[0:5])

        # Gather results
        if self.hparams.n_gpu > 1:
            global_smiles_list = [None for _ in range(self.hparams.n_gpu)]
            dist.all_gather_object(global_smiles_list, local_smiles_list)
            smiles_list = []
            for i in range(self.hparams.n_gpu):
                smiles_list += global_smiles_list[i]

            global_properties = [torch.zeros_like(local_properties) for _ in range(self.hparams.n_gpu)]
            dist.all_gather(global_properties, local_properties)
            properties = torch.cat(global_properties, dim=0)
        else:
            smiles_list = local_smiles_list
            properties = local_properties

        idx_valid = []
        valid_smiles_list = []
        valid_mols_list = []
        for smiles in smiles_list:
            if smiles is not None:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    idx_valid += [True]
                    valid_smiles_list += [smiles]
                    valid_mols_list += [mol]
                else:
                    idx_valid += [False]
            else:
                idx_valid += [False]
        unique_smiles_set = set(valid_smiles_list)
        novel_smiles_list = [smiles for smiles in valid_smiles_list if smiles not in self.train_smiles_set]
        efficient_smiles_list = [smiles for smiles in unique_smiles_set if smiles in novel_smiles_list]
        statistics = dict()

        statistics[f"sample/valid"] = float(len(valid_smiles_list)) / self.hparams.num_samples_gflownet
        statistics[f"sample/unique"] = float(len(unique_smiles_set)) / len(valid_smiles_list)
        statistics[f"sample/novel"] = float(len(novel_smiles_list)) / len(valid_smiles_list)
        statistics[f"sample/efficient"] = float(len(efficient_smiles_list)) / self.hparams.num_samples_gflownet
        if self.train_dataset.scaler_properties is not None:
            properties_unscaled = torch.tensor(self.train_dataset.scaler_properties.inverse_transform(properties[idx_valid].cpu().numpy())).to(device=self.device, dtype=properties.dtype)
        else:
            properties_unscaled = properties[idx_valid]
        print(properties_unscaled[0])

        statistics[f"gflownet/weighted_reward"], statistics[f"gflownet/weighted_diversity"], statistics[f"gflownet/mean_reward"], statistics[f"gflownet/mean_diversity"], statistics[f"gflownet/reward0"], statistics[f"gflownet/reward1"], statistics[f"gflownet/reward2"], statistics[f"gflownet/reward3"], top_10_smiles = best_rewards_gflownet(valid_smiles_list, valid_mols_list, device=self.device)
        print("Top-10 molecules")
        print(top_10_smiles)
        for key, val in statistics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)

    # Property-conditional on test properties with specific metrics
    def check_samples_dit(self):

        print("Random forest classifier for the non-rdkit property")
        model_path = f"../resource/data/{self.hparams.dataset_name}/forest_model.csv.gz"
        forest_model = TaskModel(model_path, self.train_dataset.smiles_list, self.train_dataset.properties, 
            smiles_list_valid=self.test_dataset.smiles_list, properties_valid=self.test_dataset.properties, 
            i = 0, task_type = 'classification')

        print('Intermediate statistics for Frechet distance on test set')
        stats_path = f"../resource/data/{self.hparams.dataset_name}/fcd_stats.npy"
        try:
            stat_ref = load(stats_path)
        except:
            torch.backends.cudnn.enabled = False
            stat_ref = compute_intermediate_statistics(self.test_dataset.smiles_list, n_jobs=self.hparams.num_workers-1, device=self.device, batch_size=512)
            torch.backends.cudnn.enabled = True

        num_samples = self.hparams.num_samples // self.hparams.n_gpu if not self.trainer.sanity_checking else 2
        if self.hparams.test_on_train_data:
            smiles_list = self.train_dataset.smiles_list
            properties = self.train_dataset.properties
        elif self.hparams.test_on_test_data:
            smiles_list = self.test_dataset.smiles_list
            properties = self.test_dataset.properties
        else:
            local_smiles_list, results, local_properties, _ = self.sample(num_samples)
            if self.hparams.n_gpu > 1:
                # Gather results
                global_smiles_list = [None for _ in range(self.hparams.n_gpu)]
                dist.all_gather_object(global_smiles_list, local_smiles_list)
                smiles_list = []
                for i in range(self.hparams.n_gpu):
                    smiles_list += global_smiles_list[i]

                global_properties = [torch.zeros_like(local_properties) for _ in range(self.hparams.n_gpu)]
                dist.all_gather(global_properties, local_properties)
                properties = torch.cat(global_properties, dim=0)
            else:
                smiles_list = local_smiles_list
                properties = local_properties

        print('metrics')
        idx_valid = [smiles is not None and Chem.MolFromSmiles(smiles) is not None for smiles in smiles_list]
        valid_smiles_list = [smiles for i, smiles in enumerate(smiles_list) if idx_valid]
        unique_smiles_set = set(valid_smiles_list)
        novel_smiles_list = [smiles for smiles in valid_smiles_list if smiles not in self.train_smiles_set]
        efficient_smiles_list = [smiles for smiles in unique_smiles_set if smiles in novel_smiles_list]
        statistics = dict()
        statistics["sample/valid"] = float(len(valid_smiles_list)) / self.hparams.num_samples
        statistics["sample/unique"] = float(len(unique_smiles_set)) / len(valid_smiles_list)
        statistics["sample/novel"] = float(len(novel_smiles_list)) / len(valid_smiles_list)
        statistics["sample/efficient"] = float(len(efficient_smiles_list)) / self.hparams.num_samples
        for key, val in statistics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)

        if self.train_dataset.scaler_properties is not None:
            if self.hparams.test_on_train_data or self.hparams.test_on_test_data:
                properties_unscaled = self.train_dataset.scaler_properties.inverse_transform(properties)
            else:
                properties_unscaled = self.train_dataset.scaler_properties.inverse_transform(properties.cpu().numpy())
        else:
            if self.hparams.test_on_train_data or self.hparams.test_on_test_data:
                properties_unscaled = properties
            else:
                properties_unscaled = properties.cpu().numpy()
        torch.backends.cudnn.enabled = False
        metrics = compute_molecular_metrics(task_name=self.hparams.dataset_name, molecule_list=smiles_list, targets=properties_unscaled, stat_ref=stat_ref, task_evaluator=forest_model, n_jobs=self.hparams.num_workers-1, device=self.device, batch_size=512)
        torch.backends.cudnn.enabled = True
        for key, val in metrics.items():
            self.log(key, val, on_step=False, on_epoch=True, logger=True, sync_dist=self.hparams.n_gpu > 1)

    @staticmethod
    def add_args(parser):
        #
        parser.add_argument("--dataset_name", type=str, default="zinc") # zinc, qm9, moses, chromophore, hiv, bbbp, bace

        # Options for fine-tuning
        parser.add_argument("--finetune_dataset_name", type=str, default="") # when not empty, the dataset is used for fine-tuning
        parser.add_argument("--finetune_scaler_vocab", action="store_true") # If True, uses the fine-tuning dataset vocab instead of the pre-training dataset vocab for z-score standardization

        parser.add_argument("--batch_size", type=int, default=512)
        parser.add_argument("--num_workers", type=int, default=6)
        parser.add_argument("--scaling_type", type=str, default="std") # scaling used on properties (none, std, quantile, minmax)
        parser.add_argument("--randomize_order", action="store_true") # randomize order of nodes and edges for the spanning tree to produce more diversity
        parser.add_argument("--start_random", action="store_true") # We can already randomize the order, but this also make the starting atom random

        #
        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--emb_size", type=int, default=1024)
        parser.add_argument("--nhead", type=int, default=16)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--input_dropout", type=float, default=0.0)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--lambda_predict_prop", type=float, default=0.0) # if > 0, predict all properties of the molecule
        parser.add_argument("--best_out_of_k", type=int, default=1) # If >1, we sample k molecules and choose the best out of the k based on the unconditional model of the generated mol property-prediction (IF THERE IS NO PROP-PREDICTION, we just take the first valid ones).

        parser.add_argument("--gpt", action="store_true") # use a better Transformer with Flash-attention
        parser.add_argument("--no_bias", action="store_true") # bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        parser.add_argument("--rotary", action="store_true") # rotary embedding
        parser.add_argument("--rmsnorm", action="store_true") # RMSNorm instead of LayerNorm
        parser.add_argument("--swiglu", action="store_true") # SwiGLU instead of GELU
        parser.add_argument("--expand_scale", type=float, default=2.0) # expand factor for the MLP
        parser.add_argument("--special_init", action="store_true") # the init used in GPT-2, slows down training a bit though
        parser.add_argument("--cond_lin", action="store_true") # like STGG, single linear layer for continuous variables

        #
        parser.add_argument("--disable_treeloc", action="store_true")
        parser.add_argument("--disable_graphmask", action="store_true")
        parser.add_argument("--disable_valencemask", action="store_true")
        parser.add_argument("--enable_absloc", action="store_true")
        parser.add_argument("--disable_counting_ring", action="store_true") # new
        
        #
        parser.add_argument("--lr", type=float, default=1e-3) # varies for llms
        parser.add_argument("--warmup_steps", type=int, default=200) # 200-1k should be good
        parser.add_argument("--lr_decay", type=float, default=0.1)
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.05)
        parser.add_argument("--weight_decay", type=float, default=0.1) 
        parser.add_argument("--disable_random_prop_mask", action="store_true") # new
        parser.add_argument("--not_allow_empty_bond", action="store_true") # use to disable empty bonds 

        #
        parser.add_argument("--max_len", type=int, default=250) # A bit higher to handle OOD
        parser.add_argument("--n_correct", type=int, default=20) # max_len=250 with n_correct=10 means that at len=240 we force the spanning-tree to close itself ASAP to prevent an incomplete spanning-tree
        parser.add_argument("--check_sample_every_n_epoch", type=int, default=10)
        parser.add_argument("--num_samples", type=int, default=10000)
        parser.add_argument("--num_samples_ood", type=int, default=2000)
        parser.add_argument("--sample_batch_size", type=int, default=1250)
        # Manually set OOD: need to be [max,min] for all properties, so should be 6 values
        # For Zinc based on https://arxiv.org/pdf/2208.10718, values should be 580, 84, 8.194, -3.281, 1.2861, 0.1778
        parser.add_argument("--ood_values", nargs='+', type=float, default=[0.0])
        parser.add_argument("--no_ood", action="store_true") # Do not do OOD sampling
        parser.add_argument("--only_ood", action="store_true") # Only do OOD sampling
        parser.add_argument("--no_test_step", action="store_true") # ignore test set, useful when doing --only_ood


        # Tunable knobs post-training
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--temperature_ood", type=float, default=1.0)
        parser.add_argument("--guidance", type=float, default=1.0) # 1.2 is a good value
        parser.add_argument("--guidance_ood", type=float, default=1.0) # 1.2 is a good value
        parser.add_argument("--guidance_rand", action="store_true") # if True, randomly choose a guidance between [0.5, 2]
        parser.add_argument("--top_k", type=int, default=0) # if > 0, we only select from the top-k tokens

        # Replicating the conditioning of the Gflownet https://arxiv.org/abs/2210.12765 experiments and metrics, this is just for the paper
        parser.add_argument("--gflownet", action="store_true")
        parser.add_argument("--gflownet_values", nargs='+', type=float, default=[0.5, 2.5, 1.0, 105.0]) # 4 properties and their conditioning value to maximize the reward
        parser.add_argument("--num_samples_gflownet", type=int, default=128) # same as paper

        # Only for DiT sampling
        parser.add_argument("--test_on_train_data", action="store_true") # If True, we use --test on the train data instead of fake generated data
        parser.add_argument("--test_on_test_data", action="store_true") # If True, we use --test on the test data instead of fake generated data
        
        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    CondGeneratorLightningModule.add_args(parser)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument("--save_checkpoint_dir", type=str, default="/network/scratch/j/jolicoea/AutoregressiveMolecules_checkpoints")
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--test", action="store_true")
    hparams = parser.parse_args()

    ## Add any specifics to your dataset here in terms of what to test, max_len expected, and properties (which are binary, which are continuous)
    # Note that currently, we only allow binary or continuous properties (not categorical properties with n_cat > 2)
    hparams.graph_dit_benchmark = False
    if hparams.gflownet:
        hparams.n_properties = 4
        hparams.cat_var_index = []
    elif hparams.dataset_name in ['bbbp', 'bace', 'hiv']:
        hparams.graph_dit_benchmark = True
        hparams.n_properties = 3
        if hparams.dataset_name == 'bbbp':
            assert hparams.max_len >= 200
        if hparams.dataset_name == 'bace':
            assert hparams.max_len >= 150
        if hparams.dataset_name == 'hiv':
            assert hparams.max_len >= 150
        hparams.cat_var_index = [0]
    elif hparams.dataset_name in ['zinc', 'qm9', 'moses', 'chromophore']:
        hparams.n_properties = 3
        if hparams.dataset_name == 'zinc':
            assert hparams.max_len >= 150
        if hparams.dataset_name == 'qm9':
            assert hparams.max_len >= 50
        if hparams.dataset_name == 'chromophore':
            assert hparams.max_len >= 500
        hparams.cat_var_index = []
    else:
        raise NotImplementedError()
    hparams.cont_var_index = [i for i in range(hparams.n_properties) if i not in hparams.cat_var_index]

    print('Warning: Note that for both training and metrics, results will only be reproducible when using the same number of GPUs and num_samples/sample_batch_size')
    pl.seed_everything(hparams.seed, workers=True) # use same seed, except for the dataloaders
    model = CondGeneratorLightningModule(hparams)
    if hparams.load_checkpoint_path != "":
        model.load_state_dict(torch.load(hparams.load_checkpoint_path)["state_dict"])
    if hparams.compile:
        model = torch.compile(model)

    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMWU0ZmVlZS00MzUyLTQwZjgtYWU5YS04MzE1NDE2MzhiNDAifQ==",
        project="samsung/AutoregressiveMolecules",
        source_files="**/*.py",
        tags=hparams.tag.split("_"),
        log_model_checkpoints=False,
        )
    neptune_logger.log_hyperparams(vars(hparams))
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(hparams.save_checkpoint_dir, hparams.tag), 
        save_last=True,
        enable_version_counter=False,
    )
    trainer = pl.Trainer(
        devices=hparams.n_gpu, 
        accelerator="cpu" if hparams.cpu else "gpu",
        strategy="ddp" if hparams.n_gpu > 1 else 'auto',
        precision="bf16-mixed" if hparams.bf16 else "32-true",
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback],
        gradient_clip_val=hparams.gradient_clip_val,
        log_every_n_steps=hparams.log_every_n_steps,
        limit_val_batches=0 if hparams.test or hparams.no_test_step else None,
        num_sanity_val_steps=0 if hparams.test or hparams.no_test_step else 2,
    )
    pl.seed_everything(hparams.seed + trainer.global_rank, workers=True) # different seed per worker
    trainer.fit(model, 
            ckpt_path='last')
    pl.seed_everything(hparams.seed + trainer.global_rank, workers=True) # different seed per worker
    if hparams.test:
        trainer.test(model, 
            ckpt_path='last')