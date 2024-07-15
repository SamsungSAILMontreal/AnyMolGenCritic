## Adding property-conditioning of https://arxiv.org/pdf/2210.12765

import os
import argparse
from pathlib import Path
import numpy as np
from props.properties import penalized_logp, MolLogP_smiles, qed_smiles, ExactMolWt_smiles, compute_flat_properties_nogap
from model.mxmnet import MXMNet
from data.dataset import get_cond_datasets
DATA_DIR = "../resource/data"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='zinc')
    parser.add_argument("--MAX_LEN", type=int, default=250)
    parser.add_argument("--force_vocab_redo", action="store_true") # force rebuilding the vocabulary in case of bug or something
    params = parser.parse_args()

    raw_dir = f"{DATA_DIR}/{params.dataset_name}"
    datasets = get_cond_datasets(dataset_name=params.dataset_name, raw_dir=raw_dir, randomize_order=False, 
                MAX_LEN=params.MAX_LEN, scaling_type='std', gflownet=False, 
                n_properties=3, force_vocab_redo=params.force_vocab_redo)

    properties_path = os.path.join(raw_dir, f"properties_gflownet_nogap_train.npy")
    if not os.path.exists(properties_path):
        for split, dataset in zip(['train', 'valid', 'test'], datasets):
            # No Gap
            properties = compute_flat_properties_nogap(dataset.smiles_list, device='cuda')

            properties_path = os.path.join(raw_dir, f"properties_gflownet_nogap_{split}.npy")
            if os.path.exists(properties_path):
                os.remove(properties_path)
            with open(properties_path, 'wb') as f:
                np.save(f, properties)
            print('saved no-gap')

    datasets = get_cond_datasets(dataset_name=params.dataset_name, raw_dir=raw_dir, randomize_order=False, 
                MAX_LEN=params.MAX_LEN, scaling_type='std', gflownet=True,
                n_properties=4)