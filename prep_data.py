import numpy as np
import itertools
from os import sys
import torch.nn.functional as F
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import networkx as nx
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from collections import defaultdict
from torch_geometric.data import Data
from dataset import DatasetLoader
def get_smiles_string(kmer, base_dict):
    s = []
    for base in range(0, len(kmer)):
        b = kmer[base]
        s.append(base_dict.get(b))
    smiles = ''.join(map(str, s))+'O'
    return smiles

def make_bc_smiles_dict(barcode_list, base_dict):
    barcode_smiles_dict = {}
    for bc in barcode_list:
        barcode_smiles_dict[bc] = get_smiles_string(bc, base_dict)
    return barcode_smiles_dict

def get_n_hydro(smiles):
    '''
    get number of Hs
    '''
    mol = Chem.MolFromSmiles(smiles)
    before = mol.GetNumAtoms()
    mol = Chem.AddHs(mol)
    after = mol.GetNumAtoms()
    nH = after - before
    return nH

def get_compound_graph(smiles, Atms):
    '''
    we follow the pipeline developed by Duvenaud et al. Convolutional networks on graphs for learning molecular fingerprints, Advances in neural information processing systems, 2015; pp 2224-2232
    function returns adjacency (A) and feature matrix (X)
    '''  
    mol = Chem.MolFromSmiles(smiles)    
    X = np.zeros((mol.GetNumAtoms(), len(Atms) + 4))
    #feature matrix [unique atoms, atom_degree, nH, implicit valence, aromaticity indicator] 
    A = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()))
    #adjency matrix (binary) indicating which atom is connected to each other atom
    
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
        symbol_idx = Atms.index(symbol)
        atom_degree = len(atom.GetBonds())
        implicit_valence = mol.GetAtomWithIdx(atom_idx).GetImplicitValence()
        X[atom_idx, symbol_idx] = 1
        X[atom_idx, len(Atms)] = atom_degree
        X[atom_idx, len(Atms) + 1] = get_n_hydro(symbol)
        X[atom_idx, len(Atms) + 2] = implicit_valence
        if mol.GetAtomWithIdx(atom_idx).GetIsAromatic():
                X[atom_idx, len(Atms)+3] = 1
        
        for n in (atom.GetNeighbors()):
            neighbor_atom_idx = n.GetIdx()
            A[atom_idx, neighbor_atom_idx] = 1
    
    return A, X

def pad_compound_graph(mat_list, nAtms, axis=None):
        '''
        MutliGraphCNN assumes that the number of nodes for each graph in the dataset is same.
        for graph with arbitrary size, we append all-0 rows/columns in adjacency and feature matrices and based on max graph size
        function takes in a list of matrices, and pads them to the max graph size
        assumption is that all matrices in there should be symmetric (#atoms x #atoms)
        output is a concatenated version of the padded matrices from the lsit
        '''
        assert type(mat_list) is list
        padded_matrices = []
        num_atoms = []
        for m in mat_list:
            # print(nAtms, m.shape[0], flush=True)
            num_atoms.append(nAtms)
            pad_length = nAtms - m.shape[0]
            if axis==0:
                padded_matrices += [np.pad(m, [(0,pad_length),(0,0)], mode='constant')]
            elif axis is None:
                padded_matrices += [np.pad(m, (0,pad_length), mode='constant')]
        print('largest atom: ',max(num_atoms))
        return np.vstack(padded_matrices)

def get_AX_matrix(smiles, Atms, nAtms):
        '''
        get A and X matrices from a list of SMILES strings
        '''
        A_mat_list = []
        X_mat_list = []
        for sm in smiles:
            A, X = get_compound_graph(sm, Atms)
            A_mat_list += [A]
            X_mat_list += [X]
            
        padded_A_mat = pad_compound_graph(A_mat_list, nAtms)
        padded_X_mat = pad_compound_graph(X_mat_list, nAtms, axis=0)
        
        padded_A_mat = np.split(padded_A_mat, len(smiles), axis=0)
        padded_A_mat = np.array(padded_A_mat)
        padded_X_mat = np.split(padded_X_mat, len(smiles), axis=0)
        padded_X_mat = np.array(padded_X_mat)

        return padded_A_mat, padded_X_mat  



# Main
# __________________________________________________________________________

# Load barcode data and classes
classes = np.load('data/barcode_frontier_33way_11760examples_classes.npy') # barcode labels 0-35
features = np.load('data/barcode_frontier_33way_11760examples_5features.npy') #[mean, std, min, max, median]

# Mapping between barcodes and class labels
barcode_dict = {0 : 'CAAATA', 1 : 'TCATAC', 2 : 'ATATCT', 3 : 'CTCCAC', 4 : 'ATCTAA', 5 : 'CTCAAA', 6 : 'AAATAC', 7 : 'TCCAAC', 8 : 'CAAAAC', 9 : 'ACCTCC',
10 : 'GGGTTC', 11 : 'TGATTG', 12 : 'AGAGTT', 13 : 'AGAGGA', 14 : 'ATATCA', 15 : 'TTCTGT', 16 : 'AGCCTC', 17 : 'GATACT', 18 : 'TCTCTG', 19 : 'AATCAA', 20 : 'TGGAAG',
21 : 'GCACAT', 22 : '/iSpC3/CATAC', 23 : 'T/iSpC3/ATAC',  24 : 'TC/iSpC3/TAC', 25 : 'TCA/iSpC3/AC', 26 : 'TCAT/iSpC3/C', 27 : 'TCATA/iSpC3/', 28 : '/iSpC3/CATA/iSpC3/', 29 : 'T/iSpC3/AT/iSpC3/C', 30 : 'TC/iSpC3//iSpC3/AC',
31 : 'T/iSpC3//iSpC3/TAC', 32 : 'TCA/iSpC3//iSpC3/C', 33 : 'TCAT/iSpC3//iSpC3/', 34 : 'T/iSpC3//iSpC3//iSpC3/AC', 35 : 'AA/iSpC3/CAA'}

# For now, partition data into A-B set (standard bases) and C set (contains abasic sites)
ab_classes = classes[np.where(classes <= 21)]
c_classes = classes[np.where(classes > 21)]
ab_feats = features[np.where(classes <= 21)]
c_feats = classes[np.where(classes > 21)]

# Use list of barcodes to convert class labels (0-35) to sequence labels, then convert to SMILES representation
ab_barcode_list = ['CAAATA','TCATAC','ATATCT','CTCCAC','ATCTAA','CTCAAA','AAATAC','TCCAAC','CAAAAC','ACCTCC','GGGTTC','TGATTG','AGAGTT','AGAGGA','ATATCA','TTCTGT','AGCCTC','GATACT','TCTCTG','AATCAA','TGGAAG','GCACAT']
ab_kmer_list = [barcode_dict[c] for c in ab_classes] #rept?
# print("ab_kmer_list len:", len(ab_kmer_list))
# SMILES strings for standard DNA bases
dna_base_smiles = {'A': 'OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)CC1',
            'T': 'OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1',
            'G': 'OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)CC1',
            'C': 'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1'}

barcode_smiles_dict = make_bc_smiles_dict(ab_barcode_list, dna_base_smiles)

# Convert each label in data to SMILES representation
# test_kmer_list = ['CAAATA','TCATAC','ATATCT','CTCCAC']
# test_smiles_list = [barcode_smiles_dict[i] for i in test_kmer_list]
ab_smiles_list = set([barcode_smiles_dict[i] for i in ab_kmer_list])
# for i in range(20):
#      print(ab_smiles_list[i])
# print(len(ab_smiles_list))
# Get adjacency and feature matrices for each data point
# A,X = get_AX_matrix(test_smiles_list, ['C', 'N', 'O', 'P'], 133) 
A,X = get_AX_matrix(ab_smiles_list, ['C', 'N', 'O', 'P'], 133) # zero pad to largest number of atoms

ab_feats_medians = ab_feats[:,4] # median current values for each example

print('A: ', A.shape)
print('X: ', X.shape)
print('ab_feats: ', ab_feats_medians.shape)
print(len(ab_kmer_list))

# Uncomment to save data
# np.save('A_ab.npy', A) # adjacency matrices
# np.save('X_ab.npy', X) # feature matrices
# np.save('ab_feats_medians.npy', ab_feats_medians)
# np.save('ab_kmer_labels.npy', ab_kmer_list)

data = DatasetLoader("data/", A, X, ab_feats_medians)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# _______________________________convert adjacency data if using Geometric library (incomplete)___________________________________
# # For PyG network input, need adjacency matrices in COO formation (convert sparse adjacency matrix to edge index)
# # Edge index should be [2, num_edges], start and end coordinate for each edge in each graph
# edge_indices = []
# for adj_mat in range(len(A)):
#     graph = nx.from_numpy_array(A[adj_mat])
#     adj = nx.to_scipy_sparse_array(graph).tocoo()
#     row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
#     col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
#     edge_index = torch.stack([row, col], dim=0)
#     edge_indices.append(edge_index)

# max_size = max(arr.shape[1] for arr in edge_indices)
# for i in range(len(A)):
#     edge_indices[i] = np.pad(edge_indices[i], [(0, 0), (0, max_size - edge_indices[i].shape[1])], mode='constant')
# # print(len(edge_indices))

# # # Problem - conversion to COO results in different lengths of edge indices due to dependence on number of bonds/edges
# # # Try padding edge indices to max len (zero pad at end) - not sure if will affect GCN encoding - discuss

# # edge_ids = torch.tensor(np.dstack(edge_indices), dtype=float)
# edge_ids = torch.tensor(edge_indices, dtype=float)
# # no stack: [22, 2, 294]
# # stack: [2, 294, 22]

# def get_adjacency_info(edge_id_list):
#     edge_indices_adj = []
#     for e in edge_id_list:
#         edge_indices_adj += [[1,1], [1,1]] # how to set up adjacency info? what about edge_attributes (edge_features)?
#         print("EE: ", e[0][0])
#     edge_indices_adj = torch.tensor(edge_indices_adj)
#     edge_indices_adj = edge_indices_adj.t().to(torch.long).view(2, -1)
#     return edge_indices_adj

# print(edge_ids)
# print(edge_ids.shape)

# print(len(edge_ids))
# data = Data(x=torch.tensor(X, dtype=float), edge_index=get_adjacency_info(edge_ids))
# # do we have to create like 22 datasets?
# print("num edges: ", data.num_edges)
# data.validate(raise_on_error=True)
# # print(data)