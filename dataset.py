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
import torch_geometric
from torch_geometric.data import Data, Dataset, InMemoryDataset
import os
import random
print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class DatasetLoader(InMemoryDataset):
    def __init__(self, root, A, X, edge_labels, transform=None, pre_transform=None):
    # def __init__(self, root, A, X, edge_labels, kmer_labels, transform=None, pre_transform=None):
        self.A = A
        self.X = X
        self.edge_labels = edge_labels
        # self.kmer_labels = kmer_labels
        super(DatasetLoader, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return "not_implemented.pt"

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        pass

    def process(self):
        edge_indices = []
        for adj_mat in range(len(self.A)):
            row, col = np.where(self.A[adj_mat] != 0)
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_indices.append(edge_index)
        
        data_list = []
        for i in range(len(self.X)):
            node_feats = self._get_node_features(self.X[i])
            adjacency_info = self._get_adjacency_info(edge_indices[i])
            edge_feats = self._get_edge_features()  # currently None
            label = self._get_label(self.edge_labels[i])
            # kmer_label = self._get_label(self.kmer_labels[i])
            data = Data(
                x=node_feats,
                edge_index=adjacency_info, 
                edge_attr=edge_feats,
                y=label)
                # kmer=kmer_label)
            data.validate(raise_on_error=True)
            # data_list.append(data)
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))
        # data, slices = self.collate(data_list)
        data.validate(raise_on_error=True)
        # torch.save((data, slices), self.processed_paths[0])
        # print("num edges: ", data.num_edges)

    def _get_node_features(self, node_matrix):
        return torch.tensor(node_matrix, dtype=float)

    def _get_edge_features(self):
        return None

    def _get_adjacency_info(self, edge_id_list):
        return edge_id_list

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=float)
    
    def len(self):
        return self.data.shape[0]
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))
        return data