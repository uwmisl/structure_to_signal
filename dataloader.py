import numpy as np
from os import sys
import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
import os

def write_list_to_file(filename: str, list):
    with open("test_results/" + filename + ".txt", 'w') as f:
        for line in list:
            f.write("%s\n" % line)
class DatasetLoader(InMemoryDataset):
    def __init__(self, root, A, X, edge_labels, kmer_labels, transform=None, pre_transform=None):
        self.A = A
        self.X = X
        self.edge_labels = edge_labels
        self.kmer_labels = kmer_labels
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
            edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
            edge_indices.append(edge_index)
        
        
        # edge_indices = []
        # for adj_mat in range(len(self.A)):
        #     graph = nx.from_numpy_array(self.A[adj_mat])
        #     adj = nx.to_scipy_sparse_array(graph).tocoo()
        #     row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        #     col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        #     edge_index = torch.stack([row, col], dim=0)
        #     edge_indices.append(edge_index)

        # max_size = max(arr.shape[1] for arr in edge_indices)
        # for i in range(len(self.A)):
        #     edge_indices[i] = np.pad(edge_indices[i], [(0, 0), (0, max_size - edge_indices[i].shape[1])], mode='constant')

        # edge_ids = torch.tensor(np.dstack(edge_indices), dtype=float)
        # edge_ids = torch.tensor(np.array(edge_indices), dtype=float)
        # no stack: [22, 2, 294]
        # stack: [2, 294, 22]

        # print(edge_ids)
        # print(edge_ids.shape)

        # print(len(edge_ids))
        for i in range(len(self.X)):
            node_feats = self._get_node_features(self.X[i])
            adjacency_info = self._get_adjacency_info(edge_indices[i])
            edge_feats = self._get_edge_features()  # currently None
            label = self._get_label(self.edge_labels[i])

            data = Data(
                x=node_feats,
                edge_index=adjacency_info, 
                edge_attr=edge_feats,
                y=label,
                kmer_label=f'{self.kmer_labels[i]}')
            data.validate(raise_on_error=True)
            torch.save(data, os.path.join(self.processed_dir, f'data_{i+4096}.pt'))

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
    
    def _print_machine_spec():
        print(f"Torch version: {torch.__version__}")
        print(f"Cuda available: {torch.cuda.is_available()}")
        print(f"Torch geometric version: {torch_geometric.__version__}")
