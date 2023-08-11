import numpy as np
import itertools
from os import sys
import torch.nn.functional as F
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from collections import defaultdict
from torch_geometric.data import Data
from dataloader import DatasetLoader

# 'data/template_median68pA.model'
# dna_base_smiles = {'A': 'OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)CC1',
#                    'T': 'OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1',
#                    'G': 'OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)CC1',
#                    'C': 'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1'}

def to_barcode_feats_dicts(file_path: str):
    barcode_dict_model = {}
    ab_feats_medians_model = []
    f = open(file_path, 'r')
    row = f.readlines()
    for i in range (len(row) - 1):
        line = row[i+1].split("\t")
        barcode_dict_model[i] = line[0]
        ab_feats_medians_model.append(line[1])
    f.close()
    return barcode_dict_model, ab_feats_medians_model

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
        # print('largest atom: ',max(num_atoms))
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


class DataPrepper():
    def __init__(self, barcode_dict: dict, feat_medians: dict, dna_base_smiles: dict) -> None:
        self.barcode_dict = barcode_dict
        self.feat_medians = feat_medians
        self.dna_base_smiles = dna_base_smiles

    def get_smiles_string(self, kmer: str):
        s = []
        for base in range(0, len(kmer)):
            b = kmer[base]
            s.append(self.dna_base_smiles.get(b))
        smiles = ''.join(map(str, s))+'O'
        return smiles

    def make_bc_smiles_dict(self, barcode_list):
        barcode_smiles_dict = {}
        for bc in barcode_list:
            barcode_smiles_dict[bc] = self.get_smiles_string(bc)
        return barcode_smiles_dict
    
    def process_data(self):
        ab_barcode_list_model = self.barcode_dict.values()
        barcode_smiles_dict_model = self.make_bc_smiles_dict(ab_barcode_list_model)

        ab_smiles_list_model = set([barcode_smiles_dict_model[i] for i in ab_barcode_list_model])
        A_model,X_model = get_AX_matrix(ab_smiles_list_model, ['C', 'N', 'O', 'P'], 133) # zero pad to largest number of atoms

        ab_feats_medians_dict_model = {}
        for i in range(len(ab_barcode_list_model)):
            ab_feats_medians_dict_model[i] = float(self.feat_medians[i])
        data = DatasetLoader("data/", A_model, X_model, ab_feats_medians_dict_model)
        data.process()
        print("FINISHED")


# Main
# dna_base_smiles = {'A': 'OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)CC1',
#             'T': 'OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1',
#             'G': 'OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)CC1',
#             'C': 'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1'}

# barcode_dict_model, ab_feats_medians_model = to_barcode_feats_dicts('data/template_median68pA.model')
# data_prep = DataPrepper(barcode_dict=barcode_dict_model, feat_medians=ab_feats_medians_model, dna_base_smiles=dna_base_smiles)
# data_prep.process_data()
