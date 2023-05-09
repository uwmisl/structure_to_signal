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
from torch_geometric.data import Data, Dataset
import os
import random
print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class DatasetLoader(Dataset):
    def __init__(self, root, A, X, edge_feat, transform=None, pre_transform=None):
        self.A = A
        self.X = X
        self.edge_feat = edge_feat
        super(DatasetLoader, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return "not_implemented.pt"

    @property
    def processed_file_names(self):
        return "not_implemented.pt"

    def download(self):
        pass

    def process(self):
        edge_indices = []
        for adj_mat in range(len(self.A)):
            graph = nx.from_numpy_array(self.A[adj_mat])
            adj = nx.to_scipy_sparse_array(graph).tocoo()
            row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
            col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            edge_indices.append(edge_index)

        max_size = max(arr.shape[1] for arr in edge_indices)
        for i in range(len(self.A)):
            edge_indices[i] = np.pad(edge_indices[i], [(0, 0), (0, max_size - edge_indices[i].shape[1])], mode='constant')
        # print(len(edge_indices))

        # # Problem - conversion to COO results in different lengths of edge indices due to dependence on number of bonds/edges
        # # Try padding edge indices to max len (zero pad at end) - not sure if will affect GCN encoding - discuss

        # edge_ids = torch.tensor(np.dstack(edge_indices), dtype=float)
        edge_ids = torch.tensor(edge_indices, dtype=float)
        # no stack: [22, 2, 294]
        # stack: [2, 294, 22]

        print(edge_ids)
        print(edge_ids.shape)

        print(len(edge_ids))
        for i in range(len(self.X)):
            data = Data(x=self._get_node_features(self.X[i]), edge_index=edge_ids[i]) # what about edge_attributes (edge_features) -> is that the ab_features:: How to split?
            data.validate(raise_on_error=True)
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))
            # do we have to create like 22 datasets?
        print("num edges: ", data.num_edges)
        data.validate(raise_on_error=True)

    def _get_node_features(self, node_matrix):
        return torch.tensor(node_matrix, dtype=float)

    def _get_edge_features(self):
        return None

    def _get_adjacency_info(self, edge_id_list):
        return torch.tensor(edge_id_list, dtype=torch.long)
    def len(self):
        return self.data.shape[0]
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))
        return data