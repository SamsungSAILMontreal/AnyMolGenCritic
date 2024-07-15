# Code from https://github.com/liugangcode/Graph-DiT

from rdkit import Chem, RDLogger
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
RDLogger.DisableLog('rdApp.*')
rdBase.DisableLog('rdApp.error')
from fcd_torch import FCD as FCDMetric
from moses.metrics.metrics import FragMetric, internal_diversity
from moses.metrics.utils import get_mol, mapper
from props.sascorer import calculateScore
from sklearn.metrics import accuracy_score

import re
import time
import random
random.seed(0)
import numpy as np
from multiprocessing import Pool
import torch

bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

import math
import os
import pickle
import os.path as op

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score

# Model to predict the non-rdkit properties
class TaskModel():
    """Scores based on an ECFP classifier."""
    def __init__(self, model_path, smiles_list, properties, smiles_list_valid, properties_valid, i = 0, task_type = 'classification'):
        self.task_type = [task_type]
        self.smiles_list = smiles_list
        self.smiles_list_valid = smiles_list_valid
        self.properties = properties
        self.properties_valid = properties_valid
        self.i = i # 0-th property
        self.metric_func = roc_auc_score if 'classification' in self.task_type else mean_absolute_error

        try:
            self.model = load(model_path)
            print(' evaluator loaded')
        except:
            print(' evaluator not found, training new one...')
            if 'classification' in self.task_type:
                self.model = RandomForestClassifier(random_state=0)
            elif 'regression' in self.task_type:
                self.model = RandomForestRegressor(random_state=0)
            performance = self.train()
            dump(self.model, model_path)
            print('Oracle peformance: ', performance)

    def setup_data(self, properties, smiles_list):
        y = properties[:, self.i]
        x_smiles = np.array(smiles_list)
        mask = ~np.isnan(y)
        y = y[mask]

        if 'classification' in self.task_type:
            y = y.astype(int)

        x_smiles = x_smiles[mask]
        x_fps = []
        mask = []
        for i,smiles in enumerate(x_smiles):
            mol = Chem.MolFromSmiles(smiles)
            mask.append( int(mol is not None) )
            fp = TaskModel.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            x_fps.append(fp)
        x_fps = np.concatenate(x_fps, axis=0)
        return x_fps, y

    def train(self):
        
        x_fps, y = self.setup_data(self.properties, self.smiles_list)
        self.model.fit(x_fps, y)

        perf, perf_valid = self.test()
        return perf

    def test(self):
        x_fps, y = self.setup_data(self.properties, self.smiles_list)
        if 'classification' in self.task_type:
            scores = self.model.predict_proba(x_fps)[:, 1]
            y_pred = (scores >= 0.5).astype(int)
            perf_train = (y_pred == y).sum() / len(y)
            print(f'accuracy train: {perf_train}')
        else:
            y_pred = self.model.predict(x_fps)
            perf_train = self.metric_func(y, y_pred)
            print(f'performance train: {perf_train}')

        x_fps, y = self.setup_data(self.properties_valid, self.smiles_list_valid)
        if 'classification' in self.task_type:
            scores = self.model.predict_proba(x_fps)[:, 1]
            y_pred = (scores >= 0.5).astype(int)
            perf_valid = (y_pred == y).sum() / len(y)
            print(f'accuracy valid: {perf_valid}')
        else:
            y_pred = self.model.predict(x_fps)
            perf_valid = self.metric_func(y, y_pred)
            print(f'performance valid: {perf_valid}')
        return perf_train, perf_valid


    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i,smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append( int(mol is not None) )
            fp = TaskModel.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        if 'classification' in self.task_type:
            scores = self.model.predict_proba(fps)[:, 1]
        else:
            scores = self.model.predict(fps)
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)

###### SAS Score ######
def calculateSAS(smiles_list):
    scores = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        score = calculateScore(mol)
        scores.append(score)
    return np.float32(scores)

class BasicMolecularMetrics(object):
    def __init__(self, stat_ref=None, task_evaluator=None, n_jobs=8, device='cpu', batch_size=512):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        self.stat_ref = stat_ref
        self.task_evaluator = task_evaluator

    def compute_relaxed_validity(self, generated, ensure_connected):
        valid = []
        num_components = []
        all_smiles = []
        valid_mols = []
        covered_atoms = set()
        direct_valid_count = 0
        for smile in generated:
            if smile is None:
                mol = None
            else:
                mol = Chem.MolFromSmiles(smile)
                direct_valid_flag = True if check_mol(mol, largest_connected_comp=True) is not None else False
                if direct_valid_flag:
                    direct_valid_count += 1
                if not direct_valid_flag: # Alexia: added this, just to make sure we don't correct if its already valid! Because this code can break already working molecules, I'm not sure why it does this.
                    if not ensure_connected:
                        mol_conn, _ = correct_mol(mol, connection=True)
                        mol = mol_conn if mol_conn is not None else correct_mol(mol, connection=False)[0]
                    else: # ensure fully connected
                        mol, _ = correct_mol(mol, connection=True)
                smiles = mol2smiles(mol)
                mol = get_mol(smiles)
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                num_components.append(len(mol_frags))
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                if smiles is not None and largest_mol is not None and len(smiles) > 1 and Chem.MolFromSmiles(smiles) is not None:
                    valid_mols.append(largest_mol)
                    valid.append(smiles)
                    for atom in largest_mol.GetAtoms():
                        covered_atoms.add(atom.GetSymbol())
                    all_smiles.append(smiles)
                else:
                    all_smiles.append(None)
            except Exception as e: 
                # print(f"An error occurred: {e}")
                all_smiles.append(None)
                
        return valid, len(valid) / len(generated), direct_valid_count / len(generated), np.array(num_components), all_smiles, covered_atoms

    def evaluate(self, generated, targets, ensure_connected, len_active):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid, validity, nc_validity, num_components, all_smiles, covered_atoms = self.compute_relaxed_validity(generated, ensure_connected=ensure_connected)
        nc_mu = num_components.mean() if len(num_components) > 0 else 0
        nc_min = num_components.min() if len(num_components) > 0 else 0
        nc_max = num_components.max() if len(num_components) > 0 else 0
        
        cover_str = f"Cover {len(covered_atoms)} ({len(covered_atoms)/len_active * 100:.2f}%) atoms: {covered_atoms}"
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}% (w/o correction: {nc_validity * 100 :.2f}%), cover {len(covered_atoms)} ({len(covered_atoms)/len_active * 100:.2f}%) atoms: {covered_atoms}")
        print(f"Number of connected components of {len(generated)} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}")
        perf_train, perf_test = self.task_evaluator.test()
        print(f"RandomForest Evaluator: Training performance: {perf_train} Test performance: {perf_test}")

        if validity > 0: 
            dist_metrics = {'cover_str': len(covered_atoms) ,'validity': validity, 'validity_nc': nc_validity,
                'perf_train': perf_train, 'perf_test': perf_test}
            unique = list(set(valid))
            close_pool = False
            if self.n_jobs != 1:
                pool = Pool(self.n_jobs)
                close_pool = True
            else:
                pool = 1
            valid_mols = mapper(pool)(get_mol, valid) 
            dist_metrics['interval_diversity'] = internal_diversity(valid_mols, pool, device=self.device)
            
            start_time = time.time()
            if self.stat_ref is not None:
                kwargs = {'n_jobs': pool, 'device': self.device, 'batch_size': self.batch_size}
                kwargs_fcd = {'n_jobs': self.n_jobs, 'device': self.device, 'batch_size': self.batch_size}
                try:
                    dist_metrics['sim/Frag'] = FragMetric(**kwargs)(gen=valid_mols, pref=self.stat_ref['Frag'])
                except:
                    print('error: ', 'pool', pool)
                    print('valid_mols: ', valid_mols)
                dist_metrics['dist/FCD'] = FCDMetric(**kwargs_fcd)(gen=valid, pref=self.stat_ref['FCD'])

            evaluation_list = ['target', 'sas', 'scs']

            valid_index = np.array([True if smiles else False for smiles in all_smiles])
            targets_log = {}
            for i, name in enumerate(evaluation_list):
                targets_log[f'input_{name}'] = np.array([float('nan')] * len(valid_index))
                targets_log[f'input_{name}'] = targets[:, i]
            
            targets = targets[valid_index]
            
            for i, name in enumerate(evaluation_list):
                if name == 'scs':
                        continue
                elif name == 'sas':
                    scores = calculateSAS(valid)
                else:
                    scores = self.task_evaluator(valid)
                targets_log[f'output_{name}'] = np.array([float('nan')] * len(valid_index))
                targets_log[f'output_{name}'][valid_index] = scores
                if name == 'sas':
                    dist_metrics[f'{name}/mae'] = np.mean(np.abs(scores - targets[:, i]))
                else:
                    true_y = targets[:, i]
                    predicted_labels = (scores >= 0.5).astype(int)
                    acc = (predicted_labels == true_y).sum() / len(true_y)
                    dist_metrics[f'{name}/acc'] = acc

            end_time = time.time()
            elapsed_time = end_time - start_time
            max_key_length = max(len(key) for key in dist_metrics)
            print(f'Details over {len(valid)} ({len(generated)}) valid (total) molecules, calculating metrics using {elapsed_time:.2f} s:')
            strs = ''
            for i, (key, value) in enumerate(dist_metrics.items()):
                if isinstance(value, (int, float, np.floating, np.integer)):
                    strs = strs + f'{key:>{max_key_length}}:{value:<7.4f}\t'
                if i % 4 == 3:
                    strs = strs + '\n'
            print(strs)

            if close_pool:
                pool.close()
                pool.join()
        else:
            unique = []
            dist_metrics = {}
            targets_log = None
        return unique, dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu), all_smiles, dist_metrics, targets_log

def mol2smiles(mol):
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)

def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(mol, connection=False):
    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        if connection:
            mol_conn = connect_fragments(mol)
            # if mol_conn is not None:
            mol = mol_conn
            if mol is None:
                return None, no_correct
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            try:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                queue = []
                check_idx = 0
                for b in mol.GetAtomWithIdx(idx).GetBonds():
                    type = int(b.GetBondType())
                    queue.append((b.GetIdx(), type, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                    if type == 12:
                        check_idx += 1
                queue.sort(key=lambda tup: tup[1], reverse=True)

                if queue[-1][1] == 12:
                    return None, no_correct
                elif len(queue) > 0:
                    start = queue[check_idx][2]
                    end = queue[check_idx][3]
                    t = queue[check_idx][1] - 1
                    mol.RemoveBond(start, end)
                    if t >= 1:
                        mol.AddBond(start, end, bond_dict[t])
            except Exception as e:
                # print(f"An error occurred in correction: {e}")
                return None, no_correct
    return mol, no_correct


def check_mol(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


##### connect fragements
def select_atom_with_available_valency(frag):
    atoms = list(frag.GetAtoms())
    random.shuffle(atoms)
    for atom in atoms:
        if atom.GetAtomicNum() > 1 and atom.GetImplicitValence() > 0:
            return atom

    return None

def select_atoms_with_available_valency(frag):
    return [atom for atom in frag.GetAtoms() if atom.GetAtomicNum() > 1 and atom.GetImplicitValence() > 0]

def try_to_connect_fragments(combined_mol, frag, atom1, atom2):
    # Make copies of the molecules to try the connection
    trial_combined_mol = Chem.RWMol(combined_mol)
    trial_frag = Chem.RWMol(frag)
    
    # Add the new fragment to the combined molecule with new indices
    new_indices = {atom.GetIdx(): trial_combined_mol.AddAtom(atom) for atom in trial_frag.GetAtoms()}
    
    # Add the bond between the suitable atoms from each fragment
    trial_combined_mol.AddBond(atom1.GetIdx(), new_indices[atom2.GetIdx()], Chem.BondType.SINGLE)
    
    # Adjust the hydrogen count of the connected atoms
    for atom_idx in [atom1.GetIdx(), new_indices[atom2.GetIdx()]]:
        atom = trial_combined_mol.GetAtomWithIdx(atom_idx)
        num_h = atom.GetTotalNumHs()
        atom.SetNumExplicitHs(max(0, num_h - 1))
        
    # Add bonds for the new fragment
    for bond in trial_frag.GetBonds():
        trial_combined_mol.AddBond(new_indices[bond.GetBeginAtomIdx()], new_indices[bond.GetEndAtomIdx()], bond.GetBondType())
    
    # Convert to a Mol object and try to sanitize it
    new_mol = Chem.Mol(trial_combined_mol)
    try:
        Chem.SanitizeMol(new_mol)
        return new_mol  # Return the new valid molecule
    except Chem.MolSanitizeException:
        return None  # If the molecule is not valid, return None

def connect_fragments(mol):
    # Get the separate fragments
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) < 2:
        return mol

    combined_mol = Chem.RWMol(frags[0])

    for frag in frags[1:]:
        # Select all atoms with available valency from both molecules
        atoms1 = select_atoms_with_available_valency(combined_mol)
        atoms2 = select_atoms_with_available_valency(frag)
        
        # Try to connect using all combinations of available valency atoms
        for atom1 in atoms1:
            for atom2 in atoms2:
                new_mol = try_to_connect_fragments(combined_mol, frag, atom1, atom2)
                if new_mol is not None:
                    # If a valid connection is made, update the combined molecule and break
                    combined_mol = new_mol
                    break
            else:
                # Continue if the inner loop didn't break (no valid connection found for atom1)
                continue
            # Break if the inner loop did break (valid connection found)
            break
        else:
            # If no valid connections could be made with any of the atoms, return None
            return None

    return combined_mol

#### connect fragements

def compute_molecular_metrics(task_name, molecule_list, targets, stat_ref, task_evaluator, n_jobs=8, device='cpu', batch_size=512):
    """ molecule_list: (dict) """
    ensure_connected = True
    metrics = BasicMolecularMetrics(stat_ref, task_evaluator, n_jobs=n_jobs, device=device, batch_size=batch_size)
    if task_name == 'bace':
        len_active = 8
    elif task_name == 'bbbp':
        len_active = 10 # the original paper had 9, but it missed the single molecule with a [B] atom! xD
    elif task_name == 'hiv':
        len_active = 29
    else:
        raise NotImplementedError()
    evaluated_res = metrics.evaluate(molecule_list, targets, ensure_connected, len_active)
    all_metrics = evaluated_res[-2]
    return all_metrics

if __name__ == '__main__':
    smiles_mol = 'C1CCC1'
    print("Smiles mol %s" % smiles_mol)
    chem_mol = Chem.MolFromSmiles(smiles_mol)
    print(block_mol)
