import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from dataset import DatasetLoader

# Paper model notes:
# GCN layers with exponential linear unit (ELU) activation
# Standard 1D CNN layers with relu activation 
    # average pooling of 2x2 patches of each CNN layer output matrix
# Flatten layer, converts final 1D CNN output matrix to 1D feature vector in row-wise fashion
# Input to NN layer then combine elements of NN layer output vector linearly to produce current value

# "We used the following scaling factor to determine the number of nodes in each GCN/CNN layer of our framework:
# 𝑛=16×2(𝑙−1),
# where l is the layer index of the GCN, CNN, and NN layer groups. For instance, the number of GCN layers 
# determined to yield the best performance for DNA was 4. The number of nodes for each GCN layer was 
# therefore 128, 64, 32, and 16. The same logic was applied to all other layer groups."

# e.g. "The optimal framework for DNA analysis (ATGC DNA) has four GCN layers and three CNN layers with a 
# kernel size of 10 and 8192 nodes in the NN layer"

# Something simple to get training loop tested but somewhat random for now 
class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        num_features = 8 # x = [2926, 8] = [nodes, features] so num_features = 8? 
        self.GCNconv1 = GCNConv(num_features, 64) 
        self.GCNconv2 = GCNConv(64, 32)
        self.conv1 = nn.Conv1d(32, 16, 3)
        self.fc1 = nn.Linear(16, 22) # num_classes = 22

    def forward(self, x, edge_index):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.GCNconv1(x, edge_index))
        x = F.elu(self.GCNconv2(x, edge_index))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.fc1(x)
        return x

# Trying to replicate "optimal" layer numbers/sizes from ref. paper but we should implement parameter seach function
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         self.GCNconv1 = GCNConv(num_features, 128)
#         self.GCNconv2 = GCNConv(128, 64)
#         self.GCNconv3 = GCNConv(32, 16)
#         self.GCNconv4 = GCNConv(16, )
#         self.conv1 = nn.Conv1d( , , 10)
#         self.conv2 = nn.Conv1d( , , 10)
#         self.conv3 = nn.Conv1d( , 8192, 10)
#         self.fc1 = nn.Linear(8192, 22) # num_classes = 22

#     def forward(self, x, edge_index):
#         x, edge_index = data.x, data.edge_index
#         x = F.elu(self.GCNconv1(x, edge_index))
#         x = F.elu(self.GCNconv2(x, edge_index))
#         x = F.elu(self.GCNconv3(x, edge_index))
#         x = F.elu(self.GCNconv4(x, edge_index))
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = self.fc1(x)
#         return x


def train(epochs):
    model.train()
    train_losses = []
    train_accs = []
    for epoch in range(epochs):
        # zero gradients
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(data.x, data.edge_index)
        # Calculate loss
        loss = criterion(y_pred, data.y)
        # Calculate accuracy
        # acc = 
        # Compute gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        
        train_losses.append(loss)
        train_accs.append(acc)
    return train_losses, train_accs

@torch.no_grad()
def test():
    model.eval()
    val_losses = []
    val_accs = []
    y_pred = model()
    loss = criterion(y_pred, data.y)
    # acc = 


# implement grid search for optimal hyperparameters (nummber of GNN, CNN, NN layers, CNN kernel size)
# def tune_model():
#     return history


data = torch.load('data/processed/data.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(data)
print(data[0])


model = SimpleNetwork().to(device) 

optimizer_name = "Adam"
lr = 1e-1
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
criterion = nn.MSELoss()
epochs = 50

