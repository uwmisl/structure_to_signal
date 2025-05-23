import numpy as np
import itertools
import os
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

def get_disconnected_smiles_string(kmer, base_dict):
    s = []
    for base in range(0, len(kmer)):
        b = kmer[base]
        s.append(base_dict.get(b))
    smiles = '.'.join(map(str, s))
    return smiles

def make_bc_smiles_dict(barcode_list, base_dict):
    barcode_smiles_dict = {}
    for bc in barcode_list:
        barcode_smiles_dict[bc] = get_smiles_string(bc, base_dict)
    return barcode_smiles_dict

def get_n_hydro(smiles):
    '''
    Adapted from Ding et al. Towards inferring nanopore sequencing ionic currents from nucleotide chemical structures, Nature Communications, 2021
    '''
    mol = Chem.MolFromSmiles(smiles) # Converts the SMILES string into a molecule object using the Chem.MolFromSmiles function from the RDKit library
    before = mol.GetNumAtoms() # Retrieves the number of atoms in the original molecule before adding hydrogen atoms
    mol = Chem.AddHs(mol) # Adds hydrogen atoms to the molecule using the Chem.AddHs function from the RDKit library. This function adds explicit hydrogen atoms to the molecule representation.
    after = mol.GetNumAtoms() # Retrieves the number of atoms in the modified molecule after adding hydrogen atoms
    nH = after - before # Calculates the difference between the number of atoms after adding hydrogen atoms and the number of atoms before adding hydrogen atoms. # This difference corresponds to the number of hydrogen atoms added to the molecule
    return nH

def get_compound_graph(smiles, Atms):
    '''
    Adapted from Ding et al. Towards inferring nanopore sequencing ionic currents from nucleotide chemical structures, Nature Communications, 2021
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
        Adapted from Ding et al. Towards inferring nanopore sequencing ionic currents from nucleotide chemical structures, Nature Communications, 2021
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
 

def get_AX_matrix(df, Atms, nAtms):
    '''
    Adapted from Ding et al. Towards inferring nanopore sequencing ionic currents from nucleotide chemical structures, Nature Communications, 2021
    '''
    A_mat_list = []
    X_mat_list = []
    
    # print(len(df['smiles']))

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

def process_kmer_data(kmer_source_file, smiles_type):

    df = pd.read_csv(kmer_source_file) 

    # With sugar phosphate backbone:
    backbone_base_smiles = {'A': 'Nc1ncnc2c1ncn2C3CCC(COP(=O)(O)O)O3', 
                'T': 'OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1',
                'G': 'OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)CC1',
                'C': 'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1',
                'S': 'Cc2cn(C1CCC(COP(=O)(O)O)O1)c(N)nc2=O', #Sn
                'S': 'Cn2cc(C1CCC(COP(=O)(O)O)O1)c(N)nc2=O', #Sc
                'B': 'Nc1[nH]c(=O)nc2c1ncn2C3CCC(COP(=O)(O)O)O3',
                'P': 'Nc3nc(=O)n2ccn(C1CCC(COP(=O)(O)O)O1)c2n3',
                'Z': 'Nc2[nH]c(=O)c(C1CCC(COP(=O)(O)O)O1)cc2[N+](=O)O', #Zn
                'J': 'Nc2nc(=O)nc3n(C1CCC(COP(=O)(O)O)O1)ccn23',
                'V': 'Nc1[nH]c(=O)c([N+](=O)O)cc1C2CCC(COP(=O)(O)O)O2',
                'X': 'O=c3nc2n(C1CCC(COP(=O)(O)O)O1)ccn2c(=O)[nH]3',
                'K': 'NC(=NC(N)=C1C(OC2COP(O)(=O)O)CC2)C(=C1)[N+](O)=O',
                'Z': 'NC(=O)c2cc(C1OC(COP(=O)(O)O)C(O)C1F)c(=O)[nH]c2N'} #Za 

    # Without sugar phosphate backbone
    no_backbone_smiles = {'A': 'Nc1ncnc2[nH]cnc12', 
        'T': 'Cc1c[nH]c(=O)[nH]c1=O',
        'G': 'Nc2nc1[nH]cnc1c(=O)[nH]2',
        'C': 'Nc1cc[nH]c(=O)n1',
        'S': 'Cc1c[nH]c(N)nc1=O', #Sn
        'S': 'Cn1ccc(N)nc1=O', #Sc
        'B': 'Nc1[nH]c(=O)nc2[nH]cnc12',
        'P': 'Nc2nc(=O)n1cc[nH]c1n2',
        'Z': 'Nc1[nH]c(=O)ccc1[N+](=O)O', #Zn
        'J': 'Nc1nc(=O)nc2[nH]ccn12',
        'V': 'Nc1ccc([N+](=O)O)c(=O)[nH]1',
        'X': 'O=c2nc1[nH]ccn1c(=O)[nH]2',
        'K': 'Nc1ccc([N+](=O)O)c(N)n1',
        'Z': 'NC(=O)c1ccc(=O)[nH]c1N'} #Za

    if smiles_type == backbone: xna_base_smiles = backbone_base_smiles
    else xna_base_smiles = no_backbone_smiles

    df['smiles'] = df['KXmer'].apply(lambda x: get_smiles_string(x, xna_base_smiles))

    A,X = get_AX_matrix(df, ['C', 'N', 'O', 'P', 'F'], 44)

    level_means = col_list = df['Mean level'].values.tolist()
    kmer_labels = col_list = df['KXmer'].values.tolist()

    save_dir = os.path.join('processed_XNA/data', os.path.basename(kmer_source_file).split('.')[0])

    data = DatasetLoader(save_dir, A, X, level_means, kmer_labels)
    data.process()

# XNA data
# ------------------------------------------
# kmer source files
ATGC = 'data/ATGC_r9.4.1.csv' 
B = 'data/B_r9.4.1.csv' 
J = 'data/J_r9.4.1.csv'
Kn = 'data/Kn_r9.4.1.csv'
P = 'data/P_r9.4.1.csv'
Sc = 'data/Sc_r9.4.1.csv'
Sn = 'data/Sn/Sn_r9.4.1.csv'
V = 'data/V_r9.4.1.csv'
Xt = 'data/Xt_r9.4.1.csv'
Zn = 'data/Zn_r9.4.1.csv'
Za = 'data/Za_r9.4.1.csv'
