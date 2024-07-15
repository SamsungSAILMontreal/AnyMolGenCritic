# Inspired from https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py
# Inspired from https://github.com/snap-stanford/pretrain-gnns/blob/80608723ac3aac0f7059ffa0558f082252524493/chem/loader.py#L260

import numpy as np
import networkx as nx
import torch
from copy import copy, deepcopy
from torch.nn.utils.rnn import pad_sequence
from data.dfs import dfs_successors
from collections import defaultdict
import rdkit
from rdkit import Chem
import random
from rdkit.Chem.rdchem import ChiralType, BondType, BondDir

TOKEN2BONDFEAT = {
    "-": BondType.SINGLE,
    "=": BondType.DOUBLE,
    "#": BondType.TRIPLE,
    "$": BondType.QUADRUPLE,
}
BONDFEAT2TOKEN = {val: key for key, val in TOKEN2BONDFEAT.items()}
BOND_TOKENS = [token for token in TOKEN2BONDFEAT]
BOND_ORDERS = {
    '-': 1,
    '=': 2,
    '#': 3,
    '$': 4,
}

def get_bond_token(bond):
    return BONDFEAT2TOKEN[bond.GetBondType()]

def get_bond_order(bond_token):
    return BOND_ORDERS[bond_token]

BOS_TOKEN = "[bos]"
EOS_TOKEN = "[eos]"
PAD_TOKEN = "[pad]"
MASK_TOKEN = "[mask]"
SPECIAL_TOKENS = ["[pad]", "[mask]", "[bos]", "[eos]"]
EOS_TOKEN_id = len(SPECIAL_TOKENS)-1

EMPTY_BOND_TOKEN = "[.]" # for compounds, e.g. [Na+].[Cl-]

BRANCH_START_TOKEN = "("
BRANCH_END_TOKEN = ")"
BRANCH_TOKENS = [BRANCH_START_TOKEN, BRANCH_END_TOKEN]

RING_START_TOKEN = "[bor]"
POSSIBLE_RING_IDXS = 100
RING_END_TOKENS = [f"[eor{idx}]" for idx in range(POSSIBLE_RING_IDXS)]


class SpanningTreeVocabulary:
    def __init__(self, TOKEN2ATOMFEAT, VALENCES, sort=True): # We give the dictionary of atoms used to get a vocabulary

        if sort:
            # Sort the dictionary by token to prevent problems
            mykeys = list(TOKEN2ATOMFEAT.keys())
            mykeys.sort()
            TOKEN2ATOMFEAT = {key: TOKEN2ATOMFEAT[key] for key in mykeys}
        else:
            pass # only for legacy compatibility

        self.TOKEN2ATOMFEAT = TOKEN2ATOMFEAT
        self.ATOM_TOKENS = [token for token in TOKEN2ATOMFEAT]
        self.ATOMFEAT2TOKEN = {val: key for key, val in TOKEN2ATOMFEAT.items()}
        self.TOKENS = SPECIAL_TOKENS + [EMPTY_BOND_TOKEN] + BRANCH_TOKENS + self.ATOM_TOKENS + BOND_TOKENS + [RING_START_TOKEN] + RING_END_TOKENS
        # always keep it that way
        assert self.TOKENS[0] == PAD_TOKEN
        assert self.TOKENS[2] == BOS_TOKEN
        assert self.TOKENS[EOS_TOKEN_id] == EOS_TOKEN
        self.TOKEN2ID = {token: idx for idx, token in enumerate(self.TOKENS)}

        # Get maximum valency dictionary
        self.VALENCES = VALENCES

    def get_id(self, token):
        return self.TOKEN2ID[token]

    def get_ids(self, tokens):
        return [self.TOKEN2ID[token] for token in tokens]

    def get_token(self, token_id):
        return self.TOKENS[token_id]

    def get_max_valence(self, token):
        return self.VALENCES[token]

    def smiles2molgraph(self, smiles, randomize_atom_order=False):
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol)
        Chem.RemoveStereochemistry(mol)

        if randomize_atom_order:
            ans = list(range(mol.GetNumAtoms()))
            random.shuffle(ans)
            mol=Chem.RenumberAtoms(mol,ans)

        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), token=self.ATOMFEAT2TOKEN[(atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetNumExplicitHs())])
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), token=get_bond_token(bond))

        return G

    def molgraph2smiles(self, G):
        node_tokens = nx.get_node_attributes(G, "token")
        edge_tokens = nx.get_edge_attributes(G, "token")
        edge_tokens.update({(v, u): edge_tokens[u, v] for u, v in edge_tokens})

        mol = Chem.RWMol()
        node_to_idx = dict()
        for node in G.nodes():
            atomic_num, formal_charge, num_explicit_Hs = self.TOKEN2ATOMFEAT[node_tokens[node]]
            a = Chem.Atom(atomic_num)
            a.SetFormalCharge(formal_charge)
            a.SetNumExplicitHs(num_explicit_Hs)
            if num_explicit_Hs > 0:
                a.SetNoImplicit(True)

            idx = mol.AddAtom(a)
            node_to_idx[node] = idx

        for edge in G.edges():
            token = edge_tokens[edge]
            bond_type = TOKEN2BONDFEAT[token]
            first, second = edge
            ifirst = node_to_idx[first]
            isecond = node_to_idx[second]

            mol.AddBond(ifirst, isecond, bond_type)

        smiles = Chem.MolToSmiles(mol)
        
        return smiles

    def molgraph2mol(self, G):
        node_tokens = nx.get_node_attributes(G, "token")
        edge_tokens = nx.get_edge_attributes(G, "token")
        edge_tokens.update({(v, u): edge_tokens[u, v] for u, v in edge_tokens})

        mol = Chem.RWMol()
        node_to_idx = dict()
        for node in G.nodes():
            atomic_num, formal_charge, num_explicit_Hs = self.TOKEN2ATOMFEAT[node_tokens[node]]
            a = Chem.Atom(atomic_num)
            a.SetFormalCharge(formal_charge)
            a.SetNumExplicitHs(num_explicit_Hs)
            if num_explicit_Hs > 0:
                a.SetNoImplicit(True)

            idx = mol.AddAtom(a)
            node_to_idx[node] = idx

        for edge in G.edges():
            token = edge_tokens[edge]
            bond_type = TOKEN2BONDFEAT[token]
            first, second = edge
            ifirst = node_to_idx[first]
            isecond = node_to_idx[second]

            mol.AddBond(ifirst, isecond, bond_type)
        
        return mol

def merge_vocabs(vocab, vocab2):
    TOKEN2ATOMFEAT2 = vocab2.TOKEN2ATOMFEAT
    TOKEN2ATOMFEAT_new = deepcopy(vocab.TOKEN2ATOMFEAT)
    VALENCES2 = vocab2.VALENCES
    VALENCES_new = deepcopy(vocab.VALENCES)

    for key, value in TOKEN2ATOMFEAT2.items():
        if key not in TOKEN2ATOMFEAT_new.keys():
            TOKEN2ATOMFEAT_new[key] = value
            VALENCES_new[key] = VALENCES2[key]
        else:
            if VALENCES_new[key] == 666:
                VALENCES_new[key] = VALENCES2[key]
            elif VALENCES2[key] == 666:
                pass
            else:
                VALENCES_new[key] = max(VALENCES2[key], VALENCES_new[key])

    return SpanningTreeVocabulary(TOKEN2ATOMFEAT_new, VALENCES_new)

def get_ring_end_token(idx):
    return f"[eor{idx}]"

def get_ring_end_idx(token):
    return RING_END_TOKENS.index(token)

def pad_square(squares, padding_value=0):
    max_dim = max([square.size(0) for square in squares])
    batched_squares = torch.full((len(squares), max_dim, max_dim), padding_value, dtype=torch.long)
    for idx, square in enumerate(squares):
        batched_squares[idx, : square.size(0), : square.size(1)] = square

    return batched_squares

class Data:
    def __init__(self, MAX_LEN, vocab, n_correct=0, allow_empty_bond=True, tracker_id=0): 
        self.sequence = []
        self.tokens = []
        self.node_to_token = [dict()]
        self.node_to_valence = [dict()]
        self.MAX_LEN = MAX_LEN
        self.vocab = vocab
        self.n_correct = n_correct
        self.allow_empty_bond = allow_empty_bond
        self.tracker_id = tracker_id # used to track the molecule when debugging

        #
        self._node_offset = [-1]
        self._ring_offset = -1

        #
        self.pointer_node_traj = [[]]
        self.up_loc_square = [-np.ones((self.MAX_LEN, self.MAX_LEN), dtype=int)]
        self.down_loc_square = [-np.ones((self.MAX_LEN, self.MAX_LEN), dtype=int)]

        #
        self.n_subgraph = 1 # to handle compounds which has multiple molecules

        #
        self.branch_start_nodes = []
        self.ring_to_nodes = [defaultdict(list)]

        #
        self.started = False
        self.ended = False
        self.error = None
        self.previous_empty_bond = False

        #
        self.valence_mask_traj = []
        self.graph_mask_traj = []

        #
        self.update(self.vocab.get_id(BOS_TOKEN))

    def __len__(self):
        return len(self.G.nodes())

    def update(self, id):
        token = self.vocab.get_token(id)
        if len(self.graph_mask_traj) == 0:
            if token != BOS_TOKEN:
                self.ended = True
                self.error = "add token without bos"
                return

        elif self.graph_mask_traj[-1][id]:
            self.ended = True
            print("<<------>>")
            print(id)
            print(self.vocab.TOKENS[id])
            print(self.tokens)
            self.error = "caught by graph mask"
            return

        elif self.valence_mask_traj[-1][id]:
            self.ended = True
            print("<<------>>")
            print(id)
            print(self.vocab.TOKENS[id])
            print(self.tokens)
            self.error = "caught by valency mask"
            return

        self.sequence.append(id)
        self.tokens.append(token)

        if token in (self.vocab.ATOM_TOKENS + BOND_TOKENS):
            if self.previous_empty_bond: # If after BOS or EMPTY_BOND_TOKEN
                self._node_offset += [-1]
                self.previous_empty_bond = False
            self._node_offset[-1] += 1
            new_node = copy(self._node_offset[-1])

            self.node_to_token[-1][new_node] = token

            self.up_loc_square[-1][new_node, new_node] = 0
            self.down_loc_square[-1][new_node, new_node] = 0
            if new_node > 0:
                pointer_node = self.pointer_node_traj[-1][-1]
                self.up_loc_square[-1][new_node, :new_node] = self.up_loc_square[-1][pointer_node, :new_node] + 1
                self.down_loc_square[-1][new_node, :new_node] = self.down_loc_square[-1][pointer_node, :new_node]

                self.up_loc_square[-1][:new_node, new_node] = self.up_loc_square[-1][:new_node, pointer_node]
                self.down_loc_square[-1][:new_node, new_node] = self.down_loc_square[-1][:new_node, pointer_node] + 1
            self.pointer_node_traj[-1].append(new_node)

        elif token == BRANCH_START_TOKEN:
            pointer_node = self.pointer_node_traj[-1][-1]
            self.branch_start_nodes.append(pointer_node)
            self.pointer_node_traj[-1].append(pointer_node)

        elif token == BRANCH_END_TOKEN:
            pointer_node = self.branch_start_nodes.pop()
            self.pointer_node_traj[-1].append(pointer_node)

        elif token == RING_START_TOKEN:
            pointer_node = self.pointer_node_traj[-1][-1]
            self._ring_offset += 1
            new_ring = copy(self._ring_offset)
            self.ring_to_nodes[-1][new_ring].append(pointer_node)
            self.pointer_node_traj[-1].append(pointer_node)

        elif token in RING_END_TOKENS:
            pointer_node = self.pointer_node_traj[-1][-1]
            ring = get_ring_end_idx(token)
            self.ring_to_nodes[-1][ring].append(pointer_node)
            self.pointer_node_traj[-1].append(pointer_node)

        elif token == BOS_TOKEN:
            self.started = True

        elif token == EOS_TOKEN:
            self.ended = True

        elif token == EMPTY_BOND_TOKEN:
            self.previous_empty_bond = True
            self.n_subgraph += 1
            self.pointer_node_traj += [[]]
            self.up_loc_square += [-np.ones((self.MAX_LEN, self.MAX_LEN), dtype=int)]
            self.down_loc_square += [-np.ones((self.MAX_LEN, self.MAX_LEN), dtype=int)]
            self.node_to_token += [dict()]
            self.node_to_valence += [dict()]
            self.ring_to_nodes += [defaultdict(list)]

        must_end_soon = len(self.sequence) > self.MAX_LEN - len(self.branch_start_nodes) - self.n_correct # If we are close to the MAX_LEN, we must close the string now to prevent incomplete strings
        if must_end_soon: # we must close the Spanning-tree ASAP (Note that branches must all be closed, but rings can be left open)
            if token in self.vocab.ATOM_TOKENS:
                allowed_next_tokens = []
                if not self.all_branch_closed():
                    allowed_next_tokens.append(BRANCH_END_TOKEN)
                else:
                    allowed_next_tokens.append(EOS_TOKEN)

            elif token in BOND_TOKENS:
                allowed_next_tokens = deepcopy(self.vocab.ATOM_TOKENS)# we have no choice but adding atoms because sometimes ring-end cannot be closed due to valency
                for ring in self.ring_to_nodes[-1]:
                    if len(self.ring_to_nodes[-1][ring]) == 1 and self.ring_to_nodes[-1][ring][0] != pointer_node:
                        allowed_next_tokens.append(get_ring_end_token(ring)) 

            elif token == BRANCH_START_TOKEN:
                allowed_next_tokens = BOND_TOKENS # we have no choice but adding bonds

            elif token == BRANCH_END_TOKEN:
                allowed_next_tokens = []
                if not self.all_branch_closed():
                    allowed_next_tokens.append(BRANCH_END_TOKEN)
                else:
                    allowed_next_tokens.append(EOS_TOKEN)

            elif token == RING_START_TOKEN:
                allowed_next_tokens = []
                if not self.all_branch_closed():
                    allowed_next_tokens.append(BRANCH_END_TOKEN)
                else:
                    allowed_next_tokens.append(EOS_TOKEN)

            elif token in RING_END_TOKENS:
                allowed_next_tokens = []
                if not self.all_branch_closed():
                    allowed_next_tokens.append(BRANCH_END_TOKEN)
                else:
                    allowed_next_tokens.append(EOS_TOKEN)

            elif token == BOS_TOKEN or token == EMPTY_BOND_TOKEN:
                allowed_next_tokens = self.vocab.ATOM_TOKENS # we have no choice but adding atoms

            elif token == EOS_TOKEN:
                allowed_next_tokens = []
        else: # normal behavior
            if token in self.vocab.ATOM_TOKENS:
                allowed_next_tokens = BOND_TOKENS + [BRANCH_START_TOKEN]
                if self._ring_offset < POSSIBLE_RING_IDXS - 1:
                    allowed_next_tokens.append(RING_START_TOKEN)
                if not self.all_branch_closed():
                    allowed_next_tokens.append(BRANCH_END_TOKEN)
                else:
                    allowed_next_tokens.append(EOS_TOKEN)
                    if self.allow_empty_bond:
                        allowed_next_tokens.append(EMPTY_BOND_TOKEN)

            elif token in BOND_TOKENS:
                allowed_next_tokens = deepcopy(self.vocab.ATOM_TOKENS)
                for ring in self.ring_to_nodes[-1]:
                    if len(self.ring_to_nodes[-1][ring]) == 1 and self.ring_to_nodes[-1][ring][0] != pointer_node:
                        allowed_next_tokens.append(get_ring_end_token(ring))

            elif token == BRANCH_START_TOKEN:
                allowed_next_tokens = BOND_TOKENS

            elif token == BRANCH_END_TOKEN:
                allowed_next_tokens = [BRANCH_START_TOKEN]
                if not self.all_branch_closed():
                    allowed_next_tokens.append(BRANCH_END_TOKEN)
                else:
                    allowed_next_tokens.append(EOS_TOKEN)
                    if self.allow_empty_bond:
                        allowed_next_tokens.append(EMPTY_BOND_TOKEN)

            elif token == RING_START_TOKEN:
                allowed_next_tokens = BOND_TOKENS + [BRANCH_START_TOKEN]
                if self._ring_offset < POSSIBLE_RING_IDXS - 1:
                    allowed_next_tokens.append(RING_START_TOKEN)
                if not self.all_branch_closed():
                    allowed_next_tokens.append(BRANCH_END_TOKEN)
                else:
                    allowed_next_tokens.append(EOS_TOKEN)
                    if self.allow_empty_bond:
                        allowed_next_tokens.append(EMPTY_BOND_TOKEN)

            elif token in RING_END_TOKENS:
                allowed_next_tokens = [] #[BRANCH_START_TOKEN] # Alexia: it was [], but im not sure if it was correct
                if not self.all_branch_closed():
                    allowed_next_tokens.append(BRANCH_END_TOKEN)
                else:
                    allowed_next_tokens.append(EOS_TOKEN)
                    if self.allow_empty_bond:
                        allowed_next_tokens.append(EMPTY_BOND_TOKEN)

            elif token == BOS_TOKEN or token == EMPTY_BOND_TOKEN:
                allowed_next_tokens = deepcopy(self.vocab.ATOM_TOKENS)

            elif token == EOS_TOKEN:
                allowed_next_tokens = []

        graph_mask = np.ones(len(self.vocab.TOKENS), dtype=bool)
        graph_mask[self.vocab.get_ids(allowed_next_tokens)] = False
        self.graph_mask_traj.append(graph_mask)

        # compute valency mask
        valence_mask = np.zeros(len(self.vocab.TOKENS), dtype=bool)
        if token in self.vocab.ATOM_TOKENS:

            valence = self.vocab.get_max_valence(token)
            if new_node > 0:
                if self.node_to_token[-1][pointer_node] not in BOND_TOKENS:
                    valence -= 1
                else:
                    valence -= get_bond_order(self.node_to_token[-1][pointer_node])

            self.node_to_valence[-1][new_node] = valence

            forbidden_bond_tokens = [token_ for token_ in BOND_TOKENS if get_bond_order(token_) > valence]
            valence_mask[self.vocab.get_ids(forbidden_bond_tokens)] = True

            if valence <= 1: # because need at least single-bond inside branch and single-bond after branch finish to continue 
                valence_mask[self.vocab.get_id(RING_START_TOKEN)] = True
                valence_mask[self.vocab.get_id(BRANCH_START_TOKEN)] = True

        elif token in BOND_TOKENS:
            bond_order = get_bond_order(token)
            self.node_to_valence[-1][pointer_node] -= bond_order
            
            forbidden_atom_tokens = [token_ for token_ in self.vocab.ATOM_TOKENS if self.vocab.get_max_valence(token_) < bond_order]
            valence_mask[self.vocab.get_ids(forbidden_atom_tokens)] = True
            
            forbidden_rings = [
                get_ring_end_token(ring)
                for ring in self.ring_to_nodes[-1]
                if self.node_to_valence[-1][self.ring_to_nodes[-1][ring][0]] < (bond_order - 1)
            ]
            valence_mask[self.vocab.get_ids(forbidden_rings)] = True

        elif token == BRANCH_START_TOKEN:
            valence = self.node_to_valence[-1][pointer_node]
            forbidden_bond_tokens = [token_ for token_ in BOND_TOKENS if get_bond_order(token_) > valence]
            valence_mask[self.vocab.get_ids(forbidden_bond_tokens)] = True

        elif token == BRANCH_END_TOKEN:
            if self.node_to_valence[-1][pointer_node] == 0:
                valence_mask[self.vocab.get_id(BRANCH_START_TOKEN)] = True

        elif token == RING_START_TOKEN:
            self.node_to_valence[-1][pointer_node] -= 1

            valence = self.node_to_valence[-1][pointer_node]
            forbidden_bond_tokens = [token_ for token_ in BOND_TOKENS if get_bond_order(token_) > valence]
            valence_mask[self.vocab.get_ids(forbidden_bond_tokens)] = True
            if valence <=1 : # because need at least single-bond inside branch and single-bond after branch finish to continue 
                valence_mask[self.vocab.get_id(RING_START_TOKEN)] = True
                valence_mask[self.vocab.get_id(BRANCH_START_TOKEN)] = True

        elif token in RING_END_TOKENS:
            prev_bond_order = get_bond_order(self.node_to_token[-1][pointer_node])
            ring = get_ring_end_idx(token)
            self.node_to_valence[-1][self.ring_to_nodes[-1][ring][0]] -= prev_bond_order - 1

        self.valence_mask_traj.append(valence_mask)

    def all_branch_closed(self):
        return len(self.branch_start_nodes) == 0

    def all_ring_closed(self):
        return all([(len(self.ring_to_nodes[-1][ring]) == 2) for ring in self.ring_to_nodes[-1]])

    def to_smiles(self):
        if self.error is not None:
            return None

        for idx_subgraph in range(self.n_subgraph):
            molgraph = nx.Graph()
            num_nodes = self._node_offset[idx_subgraph] + 1
            up_loc_square = self.up_loc_square[idx_subgraph][:num_nodes, :num_nodes]
            down_loc_square = self.down_loc_square[idx_subgraph][:num_nodes, :num_nodes]

            node0s, node1s = ((up_loc_square + down_loc_square) == 1).nonzero()
            node0s, node1s = node0s[node0s < node1s], node1s[node0s < node1s]

            mollinegraph = nx.Graph()
            mollinegraph.add_nodes_from(list(range(num_nodes)))
            mollinegraph.add_edges_from(zip(node0s, node1s))
            for _, ring_nodes in self.ring_to_nodes[idx_subgraph].items():
                if len(ring_nodes) == 2:
                    node0, node1 = ring_nodes
                    mollinegraph.add_edge(node0, node1)

            for node in mollinegraph.nodes():
                token = self.node_to_token[idx_subgraph][node]
                if token in self.vocab.ATOM_TOKENS:
                    molgraph.add_node(node, token=token)
                elif token in BOND_TOKENS:
                    node0, node1 = mollinegraph.neighbors(node)
                    molgraph.add_edge(node0, node1, token=token)
            if idx_subgraph == 0:
                smiles = self.vocab.molgraph2smiles(molgraph)
            else:
                smiles = smiles + "." + self.vocab.molgraph2smiles(molgraph)
        return smiles

    @staticmethod
    def from_smiles(smiles, vocab, randomize_order=False, MAX_LEN=250, start_min=True):
        molgraph = vocab.smiles2molgraph(smiles, randomize_atom_order=randomize_order)
        atom_tokens = nx.get_node_attributes(molgraph, "token")
        bond_tokens = nx.get_edge_attributes(molgraph, "token")
        bond_tokens.update({(node1, node0): val for (node0, node1), val in bond_tokens.items()})

        tokens = nx.get_node_attributes(molgraph, "token")

        mollinegraph = nx.Graph()
        for node in molgraph.nodes:
            mollinegraph.add_node(node)
        for edge in molgraph.edges:
            u, v = edge
            mollinegraph.add_node(edge)
            mollinegraph.add_edge(u, edge)
            mollinegraph.add_edge(v, edge)

        tokens = []
        seen_ring_idxs = []
        subgraphs = [molgraph.subgraph(c).copy() for c in nx.connected_components(molgraph)]
        if randomize_order:
            random.shuffle(subgraphs)

        # Alexia: Now loops over subgraphs in order to handle compounds, e.g. [Na+].[Cl-] are two subgraphs of one node
        successors = []
        starts = []
        for subgraph in subgraphs:
            if start_min: # choose one node with minimum degree
                def keyfunc(idx):
                    return subgraph.degree(idx)
                starts += [min(subgraph.nodes, key=keyfunc)]
            else: # randomize start nodes
                starts += [random.choice(list(subgraph.nodes))]
            successors += [dfs_successors(mollinegraph, source=starts[-1], randomize_neighbors=randomize_order)] # randomize edges

        #
        edges = set()
        for i in range(len(successors)):
            for n_idx, n_jdxs in successors[i].items():
                for n_jdx in n_jdxs:
                    edges.add((n_idx, n_jdx))
                    edges.add((n_jdx, n_idx))

        ring_edges = [edge for edge in mollinegraph.edges if tuple(edge) not in edges]

        node_to_ring_idx = defaultdict(list)
        for ring_idx, (atom_node, bond_node) in enumerate(ring_edges):
            node_to_ring_idx[atom_node].append(ring_idx)
            node_to_ring_idx[bond_node].append(ring_idx)

        for i in range(len(successors)):
            if i > 0:
                tokens.append(EMPTY_BOND_TOKEN)
            to_visit = [starts[i]]
            while to_visit:
                current = to_visit.pop()
                if current in [BRANCH_START_TOKEN, BRANCH_END_TOKEN]:
                    tokens.append(current)

                elif current in atom_tokens:
                    tokens.append(atom_tokens[current])

                elif current in bond_tokens:
                    tokens.append(bond_tokens[current])

                else:
                    assert False

                if current in node_to_ring_idx:
                    for ring_idx in node_to_ring_idx[current]:
                        if ring_idx not in seen_ring_idxs:
                            tokens.append(RING_START_TOKEN)
                            seen_ring_idxs.append(ring_idx)
                        else:
                            tokens.append(get_ring_end_token(seen_ring_idxs.index(ring_idx)))

                next_nodes = successors[i].get(current, [])
                if len(next_nodes) == 1:
                    to_visit.append(next_nodes[0])

                elif len(next_nodes) > 1:
                    for next_node in reversed(next_nodes):
                        to_visit.append(BRANCH_END_TOKEN)
                        to_visit.append(next_node)
                        to_visit.append(BRANCH_START_TOKEN)

        data = Data(MAX_LEN=MAX_LEN, vocab=vocab)
        for token in tokens:
            data.update(vocab.get_id(token))
            if data.error is not None:
                print(smiles)
                print("".join(data.tokens), token)
                print(data.error)

        data.update(vocab.get_id(EOS_TOKEN))

        return data

    def featurize(self):
        #
        sequence_len = len(self.sequence)
        sequence = torch.LongTensor(np.array(self.sequence))
        
        mask = (sequence == self.vocab.get_id(RING_START_TOKEN))
        count_sequence = mask.long().cumsum(dim=0)
        count_sequence = count_sequence.masked_fill(mask, 0)
            
        graph_mask_sequence = torch.tensor(np.array(self.graph_mask_traj), dtype=torch.bool)
        valency_mask_sequence = torch.tensor(np.array(self.valence_mask_traj), dtype=torch.bool)

        #
        linear_loc_square = (
            torch.abs(torch.arange(sequence_len).unsqueeze(0) - torch.arange(sequence_len).unsqueeze(1)) + 1
        )
        linear_loc_square[linear_loc_square > self.MAX_LEN] = self.MAX_LEN

        #
        up_loc_square = []
        for idx_subgraph in range(self.n_subgraph):
            pointer_node_traj_ = self.pointer_node_traj[idx_subgraph]
            up_loc_square_ = self.up_loc_square[idx_subgraph][pointer_node_traj_][:, pointer_node_traj_]
            pad_right = 1 if self.ended and idx_subgraph + 1 == self.n_subgraph else 0
            up_loc_square_ = np.pad(up_loc_square_ + 1, (1, pad_right), "constant") # pad left for BOS or new subgraph (always true), pad right for EOS (only for last subgraph)
            up_loc_square_ = torch.LongTensor(up_loc_square_)
            up_loc_square += [up_loc_square_]
        up_loc_square = torch.block_diag(*up_loc_square)
        up_loc_square[up_loc_square > self.MAX_LEN] = self.MAX_LEN

        down_loc_square = []
        for idx_subgraph in range(self.n_subgraph):
            pointer_node_traj_ = self.pointer_node_traj[idx_subgraph]
            down_loc_square_ = self.down_loc_square[idx_subgraph][pointer_node_traj_][:, pointer_node_traj_]
            pad_right = 1 if self.ended and idx_subgraph + 1 == self.n_subgraph else 0
            down_loc_square_ = np.pad(down_loc_square_ + 1, (1, pad_right), "constant") # pad left for BOS or new subgraph (always true), pad right for EOS (only for last subgraph)
            down_loc_square_ = torch.LongTensor(down_loc_square_)
            down_loc_square += [down_loc_square_]
        down_loc_square = torch.block_diag(*down_loc_square)
        down_loc_square[down_loc_square > self.MAX_LEN] = self.MAX_LEN

        assert sequence_len == up_loc_square.shape[0] and sequence_len == down_loc_square.shape[0]

        return sequence, count_sequence, graph_mask_sequence, valency_mask_sequence, linear_loc_square, up_loc_square, down_loc_square

    @staticmethod
    def collate(data_list):
        (
            sequences,
            count_sequences, 
            graph_mask_sequences,
            valency_mask_sequences,
            linear_loc_squares,
            up_loc_squares,
            down_loc_squares,
        ) = zip(*data_list)

        sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        count_sequences = pad_sequence(count_sequences, batch_first=True, padding_value=0)
        graph_mask_sequences = pad_sequence(graph_mask_sequences, batch_first=True, padding_value=0)
        valency_mask_sequences = pad_sequence(valency_mask_sequences, batch_first=True, padding_value=0)

        linear_loc_squares = pad_square(linear_loc_squares, padding_value=0)
        up_loc_squares = pad_square(up_loc_squares, padding_value=0)
        down_loc_squares = pad_square(down_loc_squares, padding_value=0)

        return (
            sequences,
            count_sequences, 
            graph_mask_sequences,
            valency_mask_sequences,
            linear_loc_squares,
            up_loc_squares,
            down_loc_squares,
        )
