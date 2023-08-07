import numpy as np
import itertools
from os import sys
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from collections import defaultdict
from torch_geometric.data import Data
from dataset import DatasetLoader
import pandas as pd

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
    get number of Hs - takes a SMILES string, converts it into a molecule object, adds hydrogen atoms to the molecule, and calculates the number of hydrogen atoms added. 
    This function provides a convenient way to determine the number of hydrogen atoms in a given molecule
    '''
    mol = Chem.MolFromSmiles(smiles) # Converts the SMILES string into a molecule object using the Chem.MolFromSmiles function from the RDKit library
    before = mol.GetNumAtoms() # Retrieves the number of atoms in the original molecule before adding hydrogen atoms
    mol = Chem.AddHs(mol) # Adds hydrogen atoms to the molecule using the Chem.AddHs function from the RDKit library. This function adds explicit hydrogen atoms to the molecule representation.
    after = mol.GetNumAtoms() # Retrieves the number of atoms in the modified molecule after adding hydrogen atoms
    nH = after - before # Calculates the difference between the number of atoms after adding hydrogen atoms and the number of atoms before adding hydrogen atoms. # This difference corresponds to the number of hydrogen atoms added to the molecule
    return nH

def get_compound_graph(smiles, Atms):
    '''
    Based on pipeline developed by Duvenaud et al. Convolutional networks on graphs for learning molecular fingerprints, Advances in neural information processing systems, 2015; pp 2224-2232
    Converts a SMILES string into a molecule object, extracts various features of each atom, and constructs the adjacency and feature matrices representing the molecular graph
    Returns adjacency (A) and feature matrix (X)
    '''  
    mol = Chem.MolFromSmiles(smiles) # converts SMILES string into molecule object   
    X = np.zeros((mol.GetNumAtoms(), len(Atms) + 4))
    #feature matrix [unique atoms, atom_degree, nH, implicit valence, aromaticity indicator] 
    A = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()))
    #adjency matrix (binary) indicating which atom is connected to each other atom
    
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx() # retrieve index of current atom
        symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol() # get atomic symbol of current atom 
        symbol_idx = Atms.index(symbol) # finds the index of the atomic symbol in the 'Atms' list
        atom_degree = len(atom.GetBonds()) # determins degree of current atom by counting number of bonds it participates in
        implicit_valence = mol.GetAtomWithIdx(atom_idx).GetImplicitValence() # retrieves implicit valence of current atom
        X[atom_idx, symbol_idx] = 1 # sets corresponding entry in the feature matrix X to 1, indicating the presence of atomic symbol in the atom
        X[atom_idx, len(Atms)] = atom_degree # sets the entry in X corresponding to the atom's degree
        X[atom_idx, len(Atms) + 1] = get_n_hydro(symbol) # gets the number of hydrogen atoms attached to current atom and sets corresponding entry in X
        X[atom_idx, len(Atms) + 2] = implicit_valence # sets entry in X corresponding to the atom's implicit valence
        if mol.GetAtomWithIdx(atom_idx).GetIsAromatic(): # checks if current atom is aromatic, if true, set entry in X to 1
                X[atom_idx, len(Atms)+3] = 1
        
        for n in (atom.GetNeighbors()): # iterates over neighboring atoms of current atom
            neighbor_atom_idx = n.GetIdx() # retrieves index of neighboring atom
            A[atom_idx, neighbor_atom_idx] = 1 # sets entry in adjacency matrix to 1 to show connection between atom and neighbor
    
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

def add_AX_columns_to_dataframe(df, Atms, nAtms):
    '''
    Add A and X matrices as columns to DataFrame containing smiles strings
    '''
    A_mat_list = []
    X_mat_list = []
    
    print(len(df['smiles']))

    for sm in df['smiles']:
        A, X = get_compound_graph(sm, Atms)
        A_mat_list += [A]
        X_mat_list += [X]

    padded_A_mat = pad_compound_graph(A_mat_list, nAtms)
    padded_X_mat = pad_compound_graph(X_mat_list, nAtms, axis=0)
    
    padded_A_mat = np.split(padded_A_mat, len(df['smiles']), axis=0)
    padded_A_mat = np.array(padded_A_mat)
    padded_X_mat = np.split(padded_X_mat, len(df['smiles']), axis=0)
    padded_X_mat = np.array(padded_X_mat)

    return padded_A_mat, padded_X_mat 


# load data
file_path = 'data/template_median68pA.model'
df = pd.read_csv(file_path, delimiter='\t')
print(df)

dna_base_smiles = {'A': 'OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)CC1',
            'T': 'OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1',
            'G': 'OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)CC1',
            'C': 'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1'}

df['smiles'] = df['kmer'].apply(lambda x: get_smiles_string(x, dna_base_smiles))
# print(df)

A,X = add_AX_columns_to_dataframe(df, ['C', 'N', 'O', 'P'], 133)
print(A.shape)
print(X.shape)

level_means = col_list = df['level_mean'].values.tolist()
print(level_means)

data = DatasetLoader("data/", A, X, level_means)
data.process()
