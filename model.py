import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GraphConv, GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import os, glob
from matplotlib import pyplot as plt
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import random

def plot(x, y1, y2, label1, label2, x_label, y_label):
    plt.plot(list(range(x)), y1, label=label1)
    plt.plot(list(range(x)), y2, label=label2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

def calculate_rmse(y, y_pred):
    return torch.sqrt(torch.mean(torch.square(y - y_pred)))

def calculate_mae(y, y_pred):
    return torch.mean(torch.abs(y - y_pred))

def load_data(file_path):
    file_list = glob.glob('data/processed/data_*')
    data_list = [torch.load(f) for f in file_list]
    return data_list

def train_test_split(data_list, split_size=0.8):
    train_size = int(split_size * len(data_list))
    test_size = len(data_list) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data_list, [train_size, test_size])
    print('num training examples: ', len(train_dataset))
    print('num testing examples: ', len(test_dataset))
    return train_dataset, test_dataset

def min_max_scale(train_dataset, test_dataset):
    # grab y values for training set
    y_values = [data.y for data in train_dataset]
    y_tensor = torch.cat(y_values)
    # calculate min and max values
    y_min = y_tensor.min()
    y_max = y_tensor.max()
    # apply scaling to training set
    for data in train_dataset:
        data.y = (data.y - y_min)/(y_max - y_min)
    # apply scaling to testing set
    for data in test_dataset:
        data.y = (data.y - y_min)/(y_max - y_min)
    return train_dataset, test_dataset
    
def noramlize(train_dataset, test_dataset):
    # grab y values for training set    
    train_target_vals = [data.y for data in train_dataset]
    targets = torch.cat(train_target_vals)
    # calculate mean and standard deviation
    target_mean = targets.mean()
    target_std = targets.std()
    # apply normalization to training set
    for data in train_dataset:
        data.y = (data.y - target_mean)/target_std
    # apply normalization to testing set
    for data in test_dataset:
        data.y = (data.y - target_mean)/target_std
    return train_dataset, test_dataset, target_mean, target_std

def create_data_loaders(train_dataset, test_dataset, batch=128):
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    return train_loader, test_loader

def train(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index.long(), batch.batch)
            loss = criterion(out, batch.y.view(-1, 1))  # Shape: [batch_size, 1]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')


def test(model, test_loader, target_mean, target_std):
    model.eval()
    with torch.no_grad():
        pred = []
        true = []
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            pred.extend(out.cpu().numpy())
            true.extend(batch.y.cpu().numpy())

    # Denormalize the predictions and true values
    pred = (pred * target_std) + target_mean
    true = (true * target_std) + target_mean

    return pred, true

class GNN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(in_features, 128) 
        self.conv2 = GraphConv(128, 64)
        self.conv3 = GraphConv(64, 32)
        self.conv4 = GraphConv(32, 16)
        self.fc = nn.Linear(16, 1) 

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)

        # Graph embedding:
        x_embed = gap(x, batch)

        # Output layer
        x = self.fc(x_embed)

        return x


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


class GNN2(nn.Module):
    def __init__(self, in_features):
        super(GNN2, self).__init__()
        self.GCNconv1 = GCNConv(in_features, 128) 
        self.GCNconv2 = GCNConv(128, 64)
        self.GCNconv3 = GCNConv(64, 32)
        self.GCNconv4 = GCNConv(32, 16)
    
        self.CNNconv1 = nn.Conv1d(16, 64, 10) 
        self.CNNconv2 = nn.Conv1d(64, 32, 10) 
        self.CNNconv3 = nn.Conv1d(32, 16, 10)

        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)  # 1D Average Pooling

        self.fc1 = nn.Linear(384, 8192)
        self.fc2 = nn.Linear(8192, 1)

        self.dropout = nn.Dropout(p=0.1)  # 10% dropout probability

    def forward(self, x, edge_index, batch):
        x = F.elu(self.GCNconv1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.GCNconv2(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.GCNconv3(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.GCNconv4(x, edge_index))
        x = self.dropout(x)
        
        # x = gmp(x, batch) # pooling before conv could help summarize/capture higher level features
        # print(x.shape[1])
        batch_size = batch.max().item() + 1  # Assuming batch is a tensor with batch assignments for each node
        
        # Reshape the tensor for 1D Convolution
        x = x.view(batch_size, -1, x.shape[1])  # Shape: [batch_size, num_nodes, num_features]

        # Apply 1D Convolutional layers with average pooling and dropout
        x = F.relu(self.CNNconv1(x.permute(0, 2, 1)))  # permute to match Conv1d input shape
        x = self.dropout(x)
        x = self.avg_pool(F.relu(self.CNNconv2(x)))
        x = self.dropout(x)
        x = self.avg_pool(F.relu(self.CNNconv3(x)))
        x = self.dropout(x)

        # Flatten the output of the CNN layer
        x = x.view(batch_size, -1)  # Shape: [batch_size, flattened_features]

        # Apply the fully connected layers with dropout
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def print_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'Parameter: {name}, Initial Value: {param.data}')


# Main
# ___________________________________________________________________________________

# load and process data
file_path = 'data/processed/'
data_list = load_data(file_path)
random.shuffle(data_list)

train_examples, test_examples = train_test_split(data_list, 0.8)
# scaled_train, scaled_test = min_max_scale(train_examples, test_examples)
# normalized_train, noramlized_test, target_mean, target_std = noramlize(train_examples, test_examples)
# train_loader, test_loader = create_data_loaders(normalized_train, noramlized_test, batch=128)
train_loader, test_loader = create_data_loaders(train_examples, test_examples, batch=128)

# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN2(in_features=8).to(device) 
model = model.double()
print(model)
print("Parameters: ", sum(p.numel() for p in model.parameters()))

# define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


# print("Initial Weights:")
# print_weights(model)

# train model
epochs = 100
train(model, train_loader, optimizer, criterion, epochs)

# print("\nUpdated Weights:")
# print_weights(model)  # Print the weights after training

# test model
# pred, true = test(model, test_loader, target_mean, target_std)

