import copy
import rdkit
from rdkit import Chem
import networkx as nx

pt = Chem.GetPeriodicTable()

def atom_to_token(atom):
    atom_idx, charge, N_H = atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetNumExplicitHs()
    token = str(pt.GetElementSymbol(atom_idx)) 
    if N_H > 0:
        token += 'H' 
        if N_H > 1:
            token += str(N_H)
    if charge != 0: 
        if charge == 1:
            token += '+'
        elif charge == -1:
            token += '-'
        elif charge > 1:
            token += '+' + str(charge)
        elif charge < -1:
            token += str(charge)
    return token, (atom_idx, charge, N_H)

# Construct the list of atoms for the dataset
def smiles_list2atoms_list(smiles_list, TOKEN2ATOMFEAT={}, VALENCES={}):
    pt = rdkit.Chem.GetPeriodicTable()

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol)
        Chem.RemoveStereochemistry(mol)

        for atom in mol.GetAtoms():
            token, (atom_idx, charge, N_H) = atom_to_token(atom)
            if token not in TOKEN2ATOMFEAT:
                TOKEN2ATOMFEAT[token] = (atom_idx, charge, N_H)
                VALENCES[token] = 666 # infinite, we don't know

    return TOKEN2ATOMFEAT, VALENCES

def get_max_valence_from_dataset(smiles_list, VALENCES={}):
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol)
        Chem.RemoveStereochemistry(mol)
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            token, _ = atom_to_token(atom)
            valence_minus_H = atom.GetTotalValence() - atom.GetNumExplicitHs()
            if token not in VALENCES or VALENCES[token] == 666:
                VALENCES[token] = valence_minus_H
            else:
                VALENCES[token] = max(VALENCES[token], valence_minus_H)
    return VALENCES

def mols_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       label=atom.GetSymbol())
                    
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       label=int(bond.GetBondTypeAsDouble()))
        
        nx_graphs.append(G)
    return nx_graphs

