import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, ConcatDataset
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from dataset import DatasetLoader
import os, glob
from matplotlib import pyplot as plt
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from scipy.stats import linregress
import itertools
import csv
import signal
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import pandas as pd
import joblib
import random

class MeanSquaredLogError(nn.Module):
    def __init__(self):
        super(MeanSquaredLogError, self).__init__()

    def forward(self, y_pred, y_true):
        log_diff = torch.log(y_pred + 1) - torch.log(y_true + 1)
        msle_loss = torch.mean(log_diff ** 2)
        return msle_loss

def plot_true_vs_predicted(true_values, predicted_values):
    true_values = np.array(true_values)  
    predicted_values = np.array(predicted_values)  

    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predicted_values, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs. Predicted Values')
    plt.grid()

    res = stats.mstats.linregress(true_values, predicted_values)
    r_squared = res.rvalue**2

    x_line = np.linspace(min(true_values), max(true_values), 100)
    y_line = res.slope * x_line + res.intercept
    plt.plot(x_line, y_line, color='red', linestyle='--', label=f'R^2 = {r_squared:.2f}')
    plt.legend()
    plt.show()

def calculate_r2(true_values, predicted_values):
    true_values = np.array(true_values)  
    predicted_values = np.array(predicted_values)  
    res = stats.mstats.linregress(true_values, predicted_values)
    r_squared = res.rvalue**2
    return r_squared

def load_data(file_path):
    file_list = glob.glob(file_path)
    data_list = [torch.load(f) for f in file_list]
    return data_list

def load_and_label_dataset(path_pattern, label):
    data_list = load_data(path_pattern)
    for ex in data_list:
        ex.dataset = label
    return data_list

def train_test_split(data_list, split_size=0.8):
    train_size = int(split_size * len(data_list))
    test_size = len(data_list) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data_list, [train_size, test_size])
    print('num training examples: ', len(train_dataset))
    print('num testing examples: ', len(test_dataset))
    return train_dataset, test_dataset

def create_data_loaders(train_dataset, test_dataset, batch=128):
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    return train_loader, test_loader

def train(model, train_loader, optimizer, criterion, epochs):
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            # out = model(batch.x, batch.edge_index.long(), batch.batch)
            out = model(batch.x, batch.edge_index.long(), batch.batch, batch.one_hot)
            loss = criterion(out, batch.y.view(-1, 1))  #[batch_size, 1]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
    return losses

def test(model, test_loader, criterion):
    model.eval()  
    losses = []
    true_values = []
    predicted_values = []
    kmer_labels = []
    
    with torch.no_grad():  
        for batch in test_loader:
            batch = batch.to(device)
            # out = model(batch.x, batch.edge_index.long(), batch.batch)
            out = model(batch.x, batch.edge_index.long(), batch.batch, batch.one_hot)
            loss = criterion(out, batch.y.view(-1, 1))  #[batch_size, 1]
            losses.append(loss.item())
            true_values.extend(batch.y.view(-1, 1).cpu().numpy())
            predicted_values.extend(out.cpu().numpy())

            for example in batch.kmer_label:
                kmer_labels.append(example)

    average_loss = sum(losses) / len(losses)
    print(f'Average Test Loss: {average_loss:.4f}')
    
    rmse = np.sqrt(mean_squared_error(np.array(true_values), np.array(predicted_values)))
    print(f'RMSE: {rmse:.4f}')

    return losses, true_values, predicted_values, rmse, kmer_labels

class CustomGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomGCNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        sqrt_deg_inv = deg.pow(-0.5)
        norm = sqrt_deg_inv[row] * sqrt_deg_inv[col]
        edge_weight = norm.view(-1, 1)
        
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
        x = self.lin(x)
        
        return x

class GNN(nn.Module):
    def __init__(self, in_features):
        super(GNN2, self).__init__()
        self.GCNconv1 = GCNConv(in_features, 128) 
        self.GCNconv2 = GCNConv(128, 64)
        self.GCNconv3 = GCNConv(64, 32)
        self.GCNconv4 = GCNConv(32, 16)
    
        self.CNNconv1 = nn.Conv1d(16, 64, 10) 
        self.CNNconv2 = nn.Conv1d(64, 32, 10) 
        self.CNNconv3 = nn.Conv1d(32, 16, 10)

        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)  

        self.fc1 = nn.Linear(384, 8192)
        self.fc2 = nn.Linear(8192, 1)

        self.dropout = nn.Dropout(p=0.1)  

    def forward(self, x, edge_index, batch):
        x = F.elu(self.GCNconv1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.GCNconv2(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.GCNconv3(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.GCNconv4(x, edge_index))
        x = self.dropout(x)
        
        batch_size = batch.max().item() + 1  
        
        x = x.view(batch_size, -1, x.shape[1])  #[batch_size, num_nodes, num_features]

        x = F.relu(self.CNNconv1(x.permute(0, 2, 1)))  
        x = self.dropout(x)
        x = self.avg_pool(F.relu(self.CNNconv2(x)))
        x = self.dropout(x)
        x = self.avg_pool(F.relu(self.CNNconv3(x)))
        x = self.dropout(x)

        x = x.view(batch_size, -1)  #[batch_size, flattened_features]

        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class GIN(nn.Module):
    def __init__(self, in_features):
        super(GIN, self).__init__()
        
        self.GINconv1 = GINConv(nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        ))
        
        self.GINconv2 = GINConv(nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ))
        
        self.GINconv3 = GINConv(nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        ))
        
        self.GINconv4 = GINConv(nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        ))
        
        self.CNNconv1 = nn.Conv1d(16, 64, 10)
        self.CNNconv2 = nn.Conv1d(64, 32, 10)
        self.CNNconv3 = nn.Conv1d(32, 16, 10)

        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(88, 8192)
        self.fc2 = nn.Linear(8192, 1)

        self.dropout = nn.Dropout(p=0.1)  

    def forward(self, x, edge_index, batch, one_hot_encodings):        
    # def forward(self, x, edge_index, batch):        
        x = F.elu(self.GINconv1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.GINconv2(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.GINconv3(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.GINconv4(x, edge_index))
        x = self.dropout(x)

        batch_size = batch.max().item() + 1  
        
        x = x.view(batch_size, -1, x.shape[1])  #[batch_size, num_nodes, num_features]

        x = F.relu(self.CNNconv1(x.permute(0, 2, 1)))  
        x = self.dropout(x)
        x = self.avg_pool(F.relu(self.CNNconv2(x)))
        x = self.dropout(x)
        x = self.avg_pool(F.relu(self.CNNconv3(x)))
        x = self.dropout(x)

        x = x.view(batch_size, -1)  #[batch_size, flattened_features]

        oh_size = one_hot_encodings.shape[0] // batch_size
        one_hot_encodings = one_hot_encodings.view(batch_size, oh_size)
        x = torch.cat((x, one_hot_encodings), dim=1)

        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def save_kmer_results(kmers, preds, trues, dataset_name, run_id, output_dir):
    df = pd.DataFrame({
        "Kmer": kmers,
        "Predicted Value": preds,
        "True Value": trues,
    })
    save_path = os.path.join(output_dir, f"{dataset_name}_results_run{run_id}.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")

def run_xna_basedrop(test_dataset, datasets, runs=1, batch_size=128, epochs=100, results_dir="XNA_basedrop_results/"):
    '''
    Runs a base dropout experiment for each unique XNA base (trained on all other XNA datasets plus canonical dataset)
    Outputs a kmer table wih model predicted values for each kmer
    '''
    
    if test_dataset == "ATGC":
        return

    print(f"\nTesting on: {test_dataset}")
    os.makedirs(results_dir, exist_ok=True)

    for run in range(runs): 
        train_data = []
        for base, data_list in datasets.items():
            if base not in ["ATGC", test_dataset]:
                train_data.extend(data_list)
        train_data.extend(datasets["ATGC"])  
        test_data = datasets[test_dataset]

        random.shuffle(train_data)
        random.shuffle(test_data)

        print(f"Run {run} — Train size: {len(train_data)} | Test size: {len(test_data)}")

        train_loader, test_loader = create_data_loaders(train_data, test_data, batch=batch_size)

        model = GIN(in_features=9).to(device).double()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()

        train_losses = train(model, train_loader, optimizer, criterion, epochs=epochs)
        tst_losses, true_values, predicted_values, rmse, kmer_labels = test(model, test_loader, criterion)
        r2 = calculate_r2(true_values, predicted_values)

        print(f"Test on {test_dataset} -> R²: {r2:.4f}, RMSE: {rmse:.4f}")

        save_kmer_results(kmer_labels, predicted_values, true_values, test_dataset, run, results_dir)


# Main
# _______________________________________________________________________________

base_names = ['ATGC', 'Zn', 'Sc', 'Sn', 'V', 'X', 'P', 'B', 'K', 'J', 'Za']
base_paths = {base: f'XNA_data/{base}/processed/data_*'
              for base in base_names}

datasets = {base: load_and_label_dataset(path, base) for base, path in base_paths.items()}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for test_dataset in base_names:
    run_xna_basedrop(test_dataset, datasets, runs=1, epochs=100)
