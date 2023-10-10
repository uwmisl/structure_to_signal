import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GraphConv, GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
# from dataset import DatasetLoader
import os, glob
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import random
from sklearn.metrics import mean_squared_error
import wandb

from torch_geometric.graphgym.cmd_args import parse_args
import yaml

class MeanSquaredLogError(nn.Module):
    def __init__(self):
        super(MeanSquaredLogError, self).__init__()

    def forward(self, y_pred, y_true):
        log_diff = torch.log(y_pred + 1) - torch.log(y_true + 1)
        msle_loss = torch.mean(log_diff ** 2)
        return msle_loss

def plot_true_vs_predicted(true_values, predicted_values, kmer_values):
    data = [[x,y, kmer] for (x,y, kmer) in zip(np.array(true_values).flatten(), np.array(predicted_values).flatten(), np.array(kmer_values).flatten())]
    table = wandb.Table(data=data, columns=["True", "Predicted", "Kmer"])
    wandb.log({"true_predict_table" : wandb.plot.scatter(table, "True", "Predicted", title="True vs. Predicted Values")})


def plot_loss_over_epochs(losses, epochs):
    data = [[x,y] for (x, y) in zip(range(1, epochs + 1), losses)]
    table = wandb.Table(data=data, columns=["Epoch", "Loss"])
    wandb.log({"loss_epoch_table" : wandb.plot.line(table, "Epoch", "Loss", title="Loss Over Epochs")})

def load_data(file_path):
    file_list = glob.glob(f'{file_path}/processed/data_*')
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
    losses = []
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
        losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
        wandb.log({"train/loss": loss})
    return losses

def test(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    losses = []
    true_values = []
    predicted_values = []
    label_values = []
    with torch.no_grad():  # Disable gradient computation during testing
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index.long(), batch.batch)
            loss = criterion(out, batch.y.view(-1, 1))  # Shape: [batch_size, 1]
            losses.append(loss.item())
            
            true_values.extend(batch.y.view(-1, 1).cpu().numpy())
            predicted_values.extend(out.cpu().numpy())
            label_values.extend(batch.kmer_label)
    average_loss = sum(losses) / len(losses)
    print(f'Average Test Loss: {average_loss:.4f}')
    
    rmse = np.sqrt(mean_squared_error(np.array(true_values), np.array(predicted_values)))
    print(f'RMSE: {rmse:.4f}')

    return losses, true_values, predicted_values, label_values

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
        self.GCNconv1 = GCNConv(in_features, 128) # TODO: Do we want configs for num layers and the properties of the layers
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

class GATsimple(nn.Module):
    def __init__(self, in_features, heads):
        super(GATsimple, self).__init__()
        self.GATconv1 = GATConv(in_features, 128, heads) 
        self.GATconv2 = GATConv(128*heads, 64, heads)
        self.GATconv3 = GATConv(64*heads, 32, heads)
        self.GATconv4 = GATConv(32*heads, 16, heads)

        self.fc1 = nn.Linear(8512, 1)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.GATconv1(x, edge_index))
        x = F.elu(self.GATconv2(x, edge_index))
        x = F.elu(self.GATconv3(x, edge_index))
        x = F.elu(self.GATconv4(x, edge_index))
        
        batch_size = batch.max().item() + 1  # Assuming batch is a tensor with batch assignments for each node
        x = x.view(batch_size, -1)  # Shape: [batch_size, flattened_features]
        x = self.fc1(x)

        return x

class GCNsimple(nn.Module):
    def __init__(self, in_features):
        super(GCNsimple, self).__init__()
        self.GATconv1 = GCNConv(in_features, 128) 
        self.GATconv2 = GCNConv(128, 64)
        self.GATconv3 = GCNConv(64, 32)
        self.GATconv4 = GCNConv(32, 16)

        self.fc1 = nn.Linear(2128, 1)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.GATconv1(x, edge_index))
        x = F.elu(self.GATconv2(x, edge_index))
        x = F.elu(self.GATconv3(x, edge_index))
        x = F.elu(self.GATconv4(x, edge_index))
        
        batch_size = batch.max().item() + 1  # Assuming batch is a tensor with batch assignments for each node
        x = x.view(batch_size, -1)  # Shape: [batch_size, flattened_features]
        x = self.fc1(x)

        return x

# Main
# ___________________________________________________________________________________
if __name__ == '__main__':
    # Load config file
    args = parse_args()
    cfg = yaml.load(open(args.cfg_file), Loader=yaml.FullLoader)
    wandb.init(
        project="MISL Structure to Signal",
        config={
            "epochs": cfg['epochs'],
            "lr": cfg['lr'],
            "data": cfg['data'],
            "train_split_size": cfg['train_split_size'],
            "batch_size": cfg['batch_size']
        }
    )

    # load and process data
    data_list = load_data(wandb.config['data'])
    random.shuffle(data_list)

    train_examples, test_examples = train_test_split(data_list, wandb.config['train_split_size'])
    # scaled_train, scaled_test = min_max_scale(train_examples, test_examples)
    # normalized_train, noramlized_test, target_mean, target_std = noramlize(train_examples, test_examples)
    # train_loader, test_loader = create_data_loaders(normalized_train, noramlized_test, batch=128)
    train_loader, test_loader = create_data_loaders(train_examples, test_examples, batch=wandb.config['batch_size'])

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN2(in_features=8).to(device) 
    model = model.double()
    print(model)
    print("Parameters: ", sum(p.numel() for p in model.parameters()))

    # define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])
    # criterion = nn.MSELoss()
    criterion = MeanSquaredLogError()

    # train model
    train_losses = train(model, train_loader, optimizer, criterion, wandb.config['epochs'])
    plot_loss_over_epochs(train_losses, wandb.config['epochs'])
    # plot_logloss_over_epochs(train_losses, epochs)

    # test model
    tst_losses, true_values, predicted_values, kmer_values = test(model, test_loader, criterion)

    plot_true_vs_predicted(true_values, predicted_values, kmer_values)

#  new dictionary for RNA encoding for smiles string, if working, throw in DNA
