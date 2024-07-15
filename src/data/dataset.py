import os
import copy
from pathlib import Path
import torch
from joblib import dump, load
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset
from data.target_data import Data as TargetData
from data.smiles import smiles_list2atoms_list, get_max_valence_from_dataset
from data.target_data import SpanningTreeVocabulary, merge_vocabs
from props.properties import penalized_logp, MolLogP_smiles, qed_smiles, ExactMolWt_smiles
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
DATA_DIR = "../resource/data"

class DummyDataset(Dataset): 
    def __init__(self):
        self.n_properties = 1

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.zeros(1,1,1).to(dtype=torch.float32), torch.zeros(self.n_properties).to(dtype=torch.float32)


def invert_permutation_numpy(permutation):
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv

def load_or_create_3prop(smiles_list, properties_path):
    if os.path.exists(properties_path):
        properties = np.load(properties_path).astype(float)
    else:
        print("Properties dataset does not exists, making it")
        molwt = np.expand_dims(np.array(ExactMolWt_smiles(smiles_list, num_workers=6)), axis=1)
        mollogp = np.expand_dims(np.array(MolLogP_smiles(smiles_list, num_workers=6)), axis=1)
        molqed = np.expand_dims(np.array(qed_smiles(smiles_list, num_workers=6)), axis=1)
        properties = np.concatenate((molwt, mollogp, molqed), axis=1) # molwt, LogP, QED
        if os.path.exists(properties_path):
            os.remove(properties_path)
        with open(properties_path, 'wb') as f:
            np.save(f, properties)
        print('Finished making the properties, saving it to file')
        properties = properties.astype(float)
    return properties

class ColumnTransformer(): # Actually working ColumnTransformer, unlike the badly designed sklearn version which change the order the variables (complete nonsense)
    
    def __init__(self, column_transformer, continuous_prop, categorical_prop):
        self.column_transformer = column_transformer
        self.continuous_prop = continuous_prop
        self.categorical_prop = categorical_prop # for now categorical is treated as is since we only have 0/1 variables, but ideally we dummy-code it

    def fit(self, X):
        self.column_transformer.fit(X[:, self.continuous_prop])

    def transform(self, X):
        Y = copy.deepcopy(X)
        Y[:, self.continuous_prop] = self.column_transformer.transform(X[:, self.continuous_prop])
        return Y

    def inverse_transform(self, X):
        Y = copy.deepcopy(X)
        Y[:, self.continuous_prop] = self.column_transformer.inverse_transform(X[:, self.continuous_prop])
        return Y

def random_data_split(n, data_dir, data, train_ratio = 0.6, valid_ratio = 0.2, test_ratio = 0.2):
    if os.path.exists(os.path.join(data_dir, f'train_idx_{data}.json')):
        with open(os.path.join(data_dir, f'train_idx_{data}.json')) as f:
            train_index = json.load(f)
        with open(os.path.join(data_dir, f'val_idx_{data}.json')) as f:
            val_index = json.load(f)
        with open(os.path.join(data_dir, f'test_idx_{data}.json')) as f:
            test_index = json.load(f)
    else:
        full_idx = list(range(n))
        train_index, test_index, _, _ = train_test_split(full_idx, full_idx, test_size=test_ratio, random_state=42)
        train_index, val_index, _, _ = train_test_split(train_index, train_index, test_size=valid_ratio/(valid_ratio+train_ratio), random_state=42)
        with open(os.path.join(data_dir, f'train_idx_{data}.json'), 'w') as f:
            json.dump(train_index, f)
        with open(os.path.join(data_dir, f'val_idx_{data}.json'), 'w') as f:
            json.dump(val_index, f)
        with open(os.path.join(data_dir, f'test_idx_{data}.json'), 'w') as f:
            json.dump(test_index, f)
    print('dataset len', n, 'train len', len(train_index), 'val len', len(val_index), 'test len', len(test_index))
    return train_index, val_index, test_index

def fixed_data_split(n, data_dir, data):
    full_idx = list(range(n))
    # Test idx
    if data == "qm9":
        if os.path.exists(os.path.join(data_dir, f'test_idx_{data}_.json')):
            with open(os.path.join(data_dir, f'test_idx_{data}_.json')) as f:
                test_index = json.load(f)
        else:
            with open(os.path.join(data_dir, f'test_idx_{data}.json')) as f:
                test_index = json.load(f)
            test_index = test_index['test_idxs']
            test_index = [int(i) for i in test_index]
            with open(os.path.join(data_dir, f'test_idx_{data}_.json'), 'w') as f:
                json.dump(test_index, f)
    else:
        with open(os.path.join(data_dir, f'test_idx_{data}.json')) as f:
            test_index = json.load(f)

    if os.path.exists(os.path.join(data_dir, f'train_idx_{data}.json')):
        with open(os.path.join(data_dir, f'train_idx_{data}.json')) as f:
            train_index = json.load(f)
        with open(os.path.join(data_dir, f'val_idx_{data}.json')) as f:
            val_index = json.load(f)
    else:
        train_index = [i for i in full_idx if i not in test_index]
        train_index, val_index, _, _ = train_test_split(train_index, train_index, test_size=0.05, random_state=42)
        with open(os.path.join(data_dir, f'train_idx_{data}.json'), 'w') as f:
            json.dump(train_index, f)
        with open(os.path.join(data_dir, f'val_idx_{data}.json'), 'w') as f:
            json.dump(val_index, f)
    print('dataset len', n, 'train len', len(train_index), 'val len', len(val_index), 'test len', len(test_index))
    return train_index, val_index, test_index

class PropCondDataset(Dataset): # molwt, LogP, QED
    def __init__(self, dataset_name, raw_dir, split, randomize_order, MAX_LEN, scaling_type = 'std', n_properties = 3, 
        gflownet=False, vocab=None, start_min=True, df=None, 
        scaler_std_properties=None, scaler_properties=None):
        self.raw_dir = raw_dir
        self.dataset_name = dataset_name
        self.randomize_order = randomize_order
        self.MAX_LEN = MAX_LEN
        self.start_min = start_min
        self.n_properties = n_properties

        if df is None:
            data_path = os.path.join(self.raw_dir, f"data.csv")
            df = pd.read_csv(data_path, sep=',')
            df = df.values

        categorical_prop = []
        if gflownet:
            assert self.dataset_name == 'qm9'
            train_index, val_index, test_index = fixed_data_split(df.shape[0], self.raw_dir, self.dataset_name) # same split as GDSS for Zinc and QM9
            # SMILES
            if split == 'train':
                self.smiles_list = df[train_index, 0].tolist()
                self.properties = np.expand_dims(df[train_index, 9].astype(float), axis=1)
            elif split == 'valid':
                self.smiles_list = df[val_index, 0].tolist()
                self.properties = np.expand_dims(df[val_index, 9].astype(float), axis=1)
            elif split == 'test':
                self.smiles_list = df[test_index, 0].tolist()
                self.properties = np.expand_dims(df[test_index, 9].astype(float), axis=1)
            else:
                raise NotImplementedError()

            properties_path = os.path.join(self.raw_dir, f"properties_gflownet_nogap_{split}.npy")
            properties_ = np.load(properties_path).astype(float)
            self.properties = np.concatenate((self.properties, properties_), axis=1)

        elif self.dataset_name in ['bbbp', 'bace', 'hiv']:
            train_index, val_index, test_index = random_data_split(df.shape[0], self.raw_dir, self.dataset_name, 0.6, 0.2, 0.2) # not random, has fixed seed 42
            categorical_prop = [0] # first variable is categorical
            if split == 'train':
                self.smiles_list = df[train_index, 1].tolist() # col 1
                self.properties = np.concatenate((df[train_index, 0:1], df[train_index, 3:5]), axis=1).astype(float)
            elif split == 'valid':
                self.smiles_list = df[val_index, 1].tolist() # col 1
                self.properties = np.concatenate((df[val_index, 0:1], df[val_index, 3:5]), axis=1).astype(float)
            elif split == 'test':
                self.smiles_list = df[test_index, 1].tolist() # col 1
                self.properties = np.concatenate((df[test_index, 0:1], df[test_index, 3:5]), axis=1).astype(float)
            else:
                raise NotImplementedError()
        elif self.dataset_name in ['zinc', 'qm9', 'chromophore']:
            if self.dataset_name in ['zinc', 'qm9']: # same split as GDSS
                train_index, val_index, test_index = fixed_data_split(df.shape[0], self.raw_dir, self.dataset_name)
            else: # from seed 42
                train_index, val_index, test_index = random_data_split(df.shape[0], self.raw_dir, self.dataset_name, 0.9, 0.05, 0.05)
            # SMILES
            if split == 'train':
                self.smiles_list = df[train_index, 0].tolist() # col 0
            elif split == 'valid':
                self.smiles_list = df[val_index, 0].tolist() # col 0
            elif split == 'test':
                self.smiles_list = df[test_index, 0].tolist() # col 0
            else:
                raise NotImplementedError()
            # Properties are not in data and have to be computed
            properties_path = os.path.join(self.raw_dir, f"properties_{split}.npy")
            self.properties = load_or_create_3prop(self.smiles_list, properties_path)

        print(self.smiles_list[0:3])
        print(self.properties[0:3])

        assert self.n_properties == self.properties.shape[1]
        continuous_prop = [i for i in range(self.n_properties) if i not in categorical_prop]

        # needed because the column transformer will change the order of properties (categorical ones will be at the end)
        self.vocab = vocab

        if scaler_std_properties is None:
            scaler_std = StandardScaler()
            self.scaler_std_properties = ColumnTransformer(scaler_std, continuous_prop, categorical_prop)
            self.scaler_std_properties.fit(self.properties)
        else:
            self.scaler_std_properties = scaler_std_properties

        self.scaling_type = scaling_type
        if scaler_properties is None:
            if scaling_type == 'std':
                self.scaler_properties = self.scaler_std_properties
            elif scaling_type == 'quantile':
                scaler = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(self.properties.shape[0] // 30, 1000), 10), subsample=1000000000, random_state=666) # from https://github.com/SamsungSAILMontreal/ForestDiffusion/blob/992b13816da004cf2411f666ac95599a7d60907b/TabDDPM/lib/data.py#L196
                self.scaler_properties = ColumnTransformer(scaler, continuous_prop, categorical_prop)

                self.scaler_properties.fit(self.properties)
            elif scaling_type == 'minmax':
                scaler = MinMaxScaler(feature_range=(-1, 1))
                self.scaler_properties = ColumnTransformer(scaler, continuous_prop, categorical_prop)
                self.scaler_properties.fit(self.properties)
            elif scaling_type == 'none':
                self.scaler_properties = None
            else:
                raise NotImplementedError()
        else:
            self.scaler_properties = scaler_properties

        properties_std = self.scaler_std_properties.transform(self.properties)
        print(properties_std[0:3])
        self.mu_prior=np.mean(properties_std, axis=0)   
        self.cov_prior=np.cov(properties_std.T)

        if scaling_type == 'std':
            self.properties = properties_std
        elif self.scaler_properties is not None:
            self.properties = self.scaler_properties.transform(self.properties)

    def update_vocab(self, vocab):
        self.vocab = vocab

    def update_smiles(self, smiles_list):
        self.smiles_list = smiles_list

    def update_properties(self, properties, scaler_std_properties=None, scaler_properties=None):
        self.properties = properties

        if scaler_std_properties is None:
            self.scaler_std_properties.fit(properties)
        else:
            self.scaler_std_properties = scaler_std_properties
        if scaler_properties is None and self.scaler_properties is not None:
            self.scaler_properties.fit(properties)
        else:
            self.scaler_properties = scaler_properties
        properties_std = self.scaler_std_properties.transform(properties)

        self.mu_prior=np.mean(properties_std, axis=0)
        self.cov_prior=np.cov(properties_std.T)

        if self.scaler_properties is not None:
            self.properties = self.scaler_properties.transform(self.properties)

    def get_mean_plus_std_property(self, idx, std):
        my_property_std = self.mu_prior[idx] + std*np.diag(self.cov_prior)[idx]

        properties = np.zeros((1, self.n_properties))
        properties[:, idx] = my_property_std
        print(properties)

        if self.scaling_type != 'std':
            properties = self.scaler_std_properties.inverse_transform(properties) # std to raw
            print(properties)
            if self.scaler_properties is not None:
                properties = self.scaler_properties.transform(properties) # raw to whatever
                print(properties)
        return properties

    # Conditional probability for the properties
    # Modified from https://github.com/nyu-dl/conditional-molecular-design-ssvae/blob/master/SSVAE.py#L161
    def sampling_conditional_property(self, yid, ytarget):

        id2 = [yid]
        id1 = np.setdiff1d(np.arange(self.n_properties), id2)
    
        mu1 = self.mu_prior[id1]
        mu2 = self.mu_prior[id2]
        
        cov11 = self.cov_prior[id1][:,id1]
        cov12 = self.cov_prior[id1][:,id2]
        cov22 = self.cov_prior[id2][:,id2]
        cov21 = self.cov_prior[id2][:,id1]
        
        cond_mu = np.transpose(mu1.T+np.matmul(cov12, np.linalg.inv(cov22)) * (ytarget-mu2))[0]
        cond_cov = cov11 - np.matmul(np.matmul(cov12, np.linalg.inv(cov22)), cov21)
        
        marginal_sampled = np.random.multivariate_normal(cond_mu, cond_cov, 1)
        
        properties = np.zeros(self.n_properties)
        properties[id1] = marginal_sampled
        properties[id2] = ytarget
        
        if self.scaling_type != 'std':
            properties_notransf = self.scaler_std_properties.inverse_transform(properties.reshape(1, -1))
            if self.scaler_properties is not None:
                properties = self.scaler_properties.transform(properties_notransf).reshape(-1)
            else:
                properties = properties_notransf.reshape(-1)

        return properties

    def sampling_property(self, num_samples, scale=1):
        
        properties = np.random.multivariate_normal(self.mu_prior, self.cov_prior*scale, num_samples)
        
        if self.scaling_type != 'std':
            properties_notransf = self.scaler_std_properties.inverse_transform(properties.reshape(1, -1))
            if self.scaler_properties is not None:
                properties = self.scaler_properties.transform(properties_notransf).reshape(-1)
            else:
                properties = properties_notransf.reshape(-1)

        return properties

    def update_vocabs_scalers(self, dset_new):
        if self.scaling_type != 'none':
            properties = self.scaler_properties.inverse_transform(self.properties)
        else:
            properties = self.properties
        self.update_properties(properties, scaler_std_properties=dset_new.scaler_std_properties, scaler_properties=dset_new.scaler_properties)
        self.update_vocab(dset_new.vocab)

    def update_scalers(self, dset_new):
        if self.scaling_type != 'none':
            properties = self.scaler_properties.inverse_transform(self.properties)
        else:
            properties = self.properties
        self.update_properties(properties, scaler_std_properties=dset_new.scaler_std_properties, scaler_properties=dset_new.scaler_properties)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        properties = self.properties[idx]
        return TargetData.from_smiles(smiles, self.vocab, randomize_order=self.randomize_order, MAX_LEN=self.MAX_LEN, start_min=self.start_min).featurize(), torch.from_numpy(properties).to(dtype=torch.float32)

def get_cond_datasets(dataset_name, raw_dir, randomize_order, MAX_LEN, scaling_type = 'std', n_properties = 3, 
    gflownet=False, start_min=True, force_vocab_redo=False, sort=True):

    vocab_train_path = os.path.join(raw_dir, f"vocab_trainval.npy")
    vocab_test_path = os.path.join(raw_dir, f"vocab_trainvaltest.npy")
    if force_vocab_redo:
        vocab_train = None
        vocab_test = None
        print("Building vocabulary from scratch")
    else:
        try:
            vocab_train = load(vocab_train_path)
            vocab_test = load(vocab_test_path)
            print("Loaded the vocabulary")
        except:
            vocab_train = None
            vocab_test = None
            print("Could not load the vocabulary; building from scratch")

    data_path = os.path.join(raw_dir, f"data.csv")
    df = pd.read_csv(data_path, sep=',')
    df = df.values

    train_dataset = PropCondDataset(dataset_name=dataset_name, raw_dir=raw_dir, split="train", randomize_order=randomize_order, 
        MAX_LEN=MAX_LEN, scaling_type=scaling_type, 
        gflownet=gflownet, n_properties=n_properties, 
        vocab=vocab_train, start_min=start_min, df=df)
    val_dataset = PropCondDataset(dataset_name=dataset_name, raw_dir=raw_dir, split="valid", randomize_order=False, 
        MAX_LEN=MAX_LEN, scaling_type=scaling_type, 
        gflownet=gflownet, n_properties=n_properties, 
        vocab=vocab_train, start_min=start_min, df=df, 
        scaler_std_properties=train_dataset.scaler_std_properties, scaler_properties=train_dataset.scaler_properties) # Need same scaler
    test_dataset = PropCondDataset(dataset_name=dataset_name, raw_dir=raw_dir, split="test", randomize_order=False, 
        MAX_LEN=MAX_LEN, scaling_type=scaling_type, 
        gflownet=gflownet, n_properties=n_properties, 
        vocab=vocab_test, start_min=start_min, df=df,
        scaler_std_properties=train_dataset.scaler_std_properties, scaler_properties=train_dataset.scaler_properties) # Need same scaler

    if vocab_train is None:
        all_smiles_list = train_dataset.smiles_list + val_dataset.smiles_list + test_dataset.smiles_list

        # Create Vocabulary of atoms (we can use train, valid, test; even if we will never generate tokens from valid and test due to the training)
        TOKEN2ATOMFEAT, VALENCES = smiles_list2atoms_list(all_smiles_list, TOKEN2ATOMFEAT={}, VALENCES={}) 
        
        # Calculate maximum Valency of atoms (we should get the valency from the training and validation data only)
        VALENCES = get_max_valence_from_dataset(train_dataset.smiles_list + val_dataset.smiles_list, VALENCES=VALENCES)
        # Make and update vocab
        vocab_train = SpanningTreeVocabulary(TOKEN2ATOMFEAT, VALENCES, sort=sort)
        dump(vocab_train, vocab_train_path)
        train_dataset.update_vocab(vocab_train)
        val_dataset.update_vocab(vocab_train)

        # Calculate maximum Valency of atoms (with train, val, and test datasets)
        VALENCES_test = get_max_valence_from_dataset(all_smiles_list, VALENCES=copy.deepcopy(VALENCES))
        # Make and update vocab
        vocab_test = SpanningTreeVocabulary(TOKEN2ATOMFEAT, VALENCES_test, sort=sort)
        dump(vocab_test, vocab_test_path)
        test_dataset.update_vocab(vocab_test)

    print(f"Atom vocabulary size: {len(train_dataset.vocab.ATOM_TOKENS)}")
    print(train_dataset.vocab.ATOM_TOKENS)
    print("Valences from train + val")
    print(train_dataset.vocab.VALENCES)
    print("Valences from train + val + test")
    print(test_dataset.vocab.VALENCES)

    return train_dataset, val_dataset, test_dataset

def merge_datasets(dset, dset2, scaler_std_properties=None, scaler_properties=None):

    dset_new = copy.deepcopy(dset)

    # merge properties
    if dset.scaling_type != 'none':
        properties = dset.scaler_properties.inverse_transform(dset.properties)
        properties2 = dset2.scaler_properties.inverse_transform(dset2.properties)
    else:
        properties = dset.properties
        properties2 = dset2.properties
    assert properties.shape[1] == properties2.shape[1]
    all_properties = np.concatenate((properties, properties2), axis=0)
    dset_new.update_properties(all_properties, scaler_std_properties=scaler_std_properties, scaler_properties=scaler_properties)

    # merge smiles
    all_smiles_list = dset.smiles_list + dset2.smiles_list
    dset_new.update_smiles(all_smiles_list)

    # merge vocabs
    vocab_merged = merge_vocabs(dset.vocab, dset2.vocab)
    dset_new.update_vocab(vocab_merged)

    return dset_new

def get_spanning_tree_from_smiles(smiles_list, randomize_order=False, MAX_LEN=250):

    # Create Vocabulary of atoms (we can use train, valid, test; even if we will never generate tokens from valid and test due to the training)
    TOKEN2ATOMFEAT, VALENCES = smiles_list2atoms_list(smiles_list, TOKEN2ATOMFEAT={}, VALENCES={}) 
    
    # Calculate maximum Valency of atoms (we should get the valency from the training and validation data only)
    VALENCES = get_max_valence_from_dataset(smiles_list, VALENCES=VALENCES)
    # Make and update vocab
    vocab = SpanningTreeVocabulary(TOKEN2ATOMFEAT, VALENCES)

    out = []
    for smile in smiles_list:
        out += ["".join(TargetData.from_smiles(smile, vocab, randomize_order=randomize_order, MAX_LEN=MAX_LEN, start_min=not randomize_order).tokens)]

    if len(out) == 1:
        return out[0]
    else:
        return out