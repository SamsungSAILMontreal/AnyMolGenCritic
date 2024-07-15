# Adapted from https://github.com/wengong-jin/hgraph2graph/

import torch
import rdkit
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdmolops, Descriptors, rdMolDescriptors 
from rdkit.Chem import Descriptors
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed
import networkx as nx
import props.sascorer as sascorer
from moses.utils import mapper
import torch_geometric.data as gd
from rdkit.rdBase import BlockLogs
from model import mxmnet
import numpy as np

def similarity(a, b):
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def penalized_logp(s):
    mol = Chem.MolFromSmiles(s)

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle


def MolLogP_mols_(mol):
    RDLogger.DisableLog('rdApp.*')
    return MolLogP(mol)
def MolLogP_mols(mols, num_workers=6):
    return [MolLogP_mols_(mol) for mol in mols]

def qed_mols_(mol):
    RDLogger.DisableLog('rdApp.*')
    return qed(mol)
def qed_mols(mols, num_workers=6):
    return [qed_mols_(mol) for mol in mols]

def ExactMolWt_mols_(mol):
    RDLogger.DisableLog('rdApp.*')
    return Descriptors.ExactMolWt(mol)
def ExactMolWt_mols(mols, num_workers=6):
    return [ExactMolWt_mols_(mol) for mol in mols]


def MolLogP_smiles(smiles, num_workers=6):
    return [MolLogP_mols_(Chem.MolFromSmiles(smile)) for smile in smiles]
def qed_smiles(smiles, num_workers=6):
    return [qed_mols_(Chem.MolFromSmiles(smile)) for smile in smiles]
def ExactMolWt_smiles(smiles, num_workers=6):
    return [ExactMolWt_mols_(Chem.MolFromSmiles(smile)) for smile in smiles]

def MAE_properties(mols, properties, properties_idx=[0,1,2]): # molwt, LogP, QED

    gen_properties = None

    if 0 in properties_idx:
        gen_molwt = ExactMolWt_mols(mols)
        gen_molwt = torch.tensor(gen_molwt).to(dtype=properties.dtype, device=properties.device).unsqueeze(1)
        gen_properties = gen_molwt

    if 1 in properties_idx:
        gen_logp = MolLogP_mols(mols)
        gen_logp = torch.tensor(gen_logp).to(dtype=properties.dtype, device=properties.device).unsqueeze(1)
        if gen_properties is None:
            gen_properties = gen_logp
        else:
            gen_properties = torch.cat([gen_properties, gen_logp], dim=1)

    if 2 in properties_idx:
        gen_qed = qed_mols(mols)
        gen_qed = torch.tensor(gen_qed).to(dtype=properties.dtype, device=properties.device).unsqueeze(1)
        if gen_properties is None:
            gen_properties = gen_qed
        else:
            gen_properties = torch.cat([gen_properties, gen_qed], dim=1)

    losses = (gen_properties-properties).abs().mean(dim=1)
    n = gen_properties.shape[0]
    losses_1, index_best = torch.topk(losses, 1, dim=0, largest=False, sorted=True)
    losses_10 = torch.topk(losses, min(10, n), dim=0, largest=False, sorted=True)[0]
    losses_100 = torch.topk(losses, min(100, n), dim=0, largest=False, sorted=True)[0]
    if n < 100:
        print(f'Warning: Less than 100 valid molecules ({n} valid molecules), so results will be poor')
    Min_MAE = losses_1[0]
    print(Chem.MolToSmiles(mols[index_best])) # best molecule
    Min10_MAE = torch.sum(losses_10)/10
    Min100_MAE = torch.sum(losses_100)/100
    
    return Min_MAE, Min10_MAE, Min100_MAE


# From https://arxiv.org/pdf/2210.12765

def compute_other_flat_rewards(mol):
    logp = np.exp(-((Descriptors.MolLogP(mol) - 2.5)**2) / 2)
    sa = (10 - sascorer.calculateScore(mol)) / 9  # Turn into a [0-1] reward
    molwt = np.exp(-((Descriptors.MolWt(mol) - 105)**2) / 150)
    return logp, sa, molwt

@torch.no_grad()
def compute_flat_rewards(mols, device, gap_model=None):
    assert len(mols) <= 128

    if gap_model is None:
        gap_model = mxmnet.MXMNet(mxmnet.Config(128, 6, 5.0))
        state_dict = torch.load('../resource/data/mxmnet_gap_model.pt')
        gap_model.load_state_dict(state_dict)
        gap_model.to(device)

    other_flats = torch.as_tensor(
        [compute_other_flat_rewards(mol) for mol in mols]).float().to(device)

    graphs = [mxmnet.mol2graph(i) for i in mols]
    is_valid = [graph is not None for graph in graphs]
    graphs = [graph for graph in graphs if graph is not None]
    batch = gd.Batch.from_data_list(graphs)
    batch.to(device)
    preds = gap_model(batch).reshape((-1,)).data.cpu() / mxmnet.HAR2EV
    preds[preds.isnan()] = 0 # they used 1 in gflownet, but this makes no sense and it could inflate the predicted property
    preds_mxnet = torch.as_tensor(preds).float().to(device).clip(1e-4, 2).reshape((-1, 1))
    flat_rewards = torch.cat([preds_mxnet, other_flats[is_valid]], 1)
    return flat_rewards

def compute_other_flat_properties(mol):
    logp = Descriptors.MolLogP(mol)
    sa = sascorer.calculateScore(mol)
    molwt = Descriptors.MolWt(mol)
    return logp, sa, molwt

@torch.no_grad()
def compute_flat_properties_nogap(smiles, device, num_workers=24):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    flat_properties = torch.as_tensor(
        [compute_other_flat_properties(mol) for mol in mols]).float().to(device)
    return flat_properties.cpu().detach().numpy()

# Taken from https://github.com/recursionpharma/gflownet/blob/f106cdeb6892214cbb528a3e06f4c721f4003175/src/gflownet/utils/metrics.py#L584
def top_k_diversity(mols, rewards, K=10):
    fps = [Chem.RDKFingerprint(mol) for mol in mols]
    x = []
    for i in np.argsort(rewards)[::-1]:
        y = fps[i]
        if y is None:
            continue
        x.append(y)
        if len(x) >= K:
            break
    s = np.array([DataStructs.BulkTanimotoSimilarity(i, x) for i in x])
    return (np.sum(s) - len(x)) / (len(x) * len(x) - len(x))  # substract the diagonal

def best_rewards_gflownet(smiles, mols, device): # molwt, LogP, QED

    gap_model = mxmnet.MXMNet(mxmnet.Config(128, 6, 5.0))
    state_dict = torch.load('../resource/data/mxmnet_gap_model.pt')
    gap_model.load_state_dict(state_dict)
    gap_model.to(device)

    gen_rewards = None
    offset = 0
    num_samples = len(mols)
    while offset < num_samples:
        cur_num_samples = min(num_samples - offset, 128)
        gen_rewards_ = compute_flat_rewards(mols[offset:(offset+cur_num_samples)], device, gap_model=gap_model)
        offset += cur_num_samples
        print(offset)
        if gen_rewards is None:
            gen_rewards = gen_rewards_
        else:
            gen_rewards = torch.cat((gen_rewards_, gen_rewards), dim=0)
    gen_rewards_mean = gen_rewards.mean(dim=1)
    n = gen_rewards.shape[0]
    rewards_10, indexes_10 = torch.topk(gen_rewards_mean, min(10, n), dim=0, largest=True, sorted=True)
    top_rewards = gen_rewards[indexes_10, :].mean(0)
    top_rewards_mean = top_rewards.mean()
    diversity = top_k_diversity(mols, gen_rewards_mean.cpu().numpy(), K=10)

    top_rewards_weighted = 0
    diversity_weighted = 0
    for i in range(10):
        torch.manual_seed(i) # deterministic
        w = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.5] * 4)).sample().to(device)
        gen_rewards_weighted = (w*gen_rewards).sum(dim=1)
        n = gen_rewards.shape[0]
        rewards_10, indexes_10_ = torch.topk(gen_rewards_weighted, min(10, n), dim=0, largest=True, sorted=True)
        top_rewards_weighted += rewards_10.mean() / 10
        diversity_weighted += top_k_diversity(mols, gen_rewards_weighted.cpu().numpy(), K=10) / 10

    return top_rewards_weighted, diversity_weighted, top_rewards_mean, diversity, top_rewards[0], top_rewards[1], top_rewards[2], top_rewards[3], [smiles[i] for i in indexes_10]


if __name__ == "__main__":
    print(
        round(
            penalized_logp("ClC1=CC=C2C(C=C(C(C)=O)C(C(NC3=CC(NC(NC4=CC(C5=C(C)C=CC=C5)=CC=C4)=O)=CC=C3)=O)=C2)=C1"), 2
        ),
        5.30,
    )
