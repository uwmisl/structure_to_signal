import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from dataset import DatasetLoader
import os, glob
from matplotlib import pyplot as plt
import numpy as np

def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

def plot_train_test(epochs, train_losses, test_losses):
    plt.plot(list(range(epochs)), train_losses, label="Train")
    plt.plot(list(range(epochs)), test_losses, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

class SimpleGNN(nn.Module):
    def __init__(self):
        super(SimpleGNN, self).__init__()
        torch.manual_seed(42)

        self.conv1 = GCNConv(num_features, embedding_size) 
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        self.conv4 = GCNConv(embedding_size, embedding_size)
        self.out = nn.Linear(embedding_size*2, 1) 

    def forward(self, x, edge_index, batch_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.conv3(h, edge_index)
        h = F.relu(h)
        h = self.conv4(h, edge_index)
        h = F.relu(h)
          
        # Global Pooling (stack mean pooling and max pooling)
        h = torch.cat([gmp(h, batch_index), 
                            gap(h, batch_index)], dim=1)

        # Output layer
        out = self.out(h)

        return out, h

def train(train_loader):#, model, optimizer, criterion):
    model.train()
    for train_data in train_loader:
        # zero gradients
        optimizer.zero_grad()
        # Forward pass
        y_pred, embedding = model(train_data.x, train_data.edge_index.long(), train_data.batch)
        # Calculate loss
        # print(f'y: {train_data.y.shape} pred: {y_pred.shape}')
        loss = criterion(y_pred, train_data.y.unsqueeze(1))
        # Compute gradients
        loss.backward()
        # Update parameters
        optimizer.step()
    return loss, embedding

@torch.no_grad()
def test(test_loader):#, model, optimizer, criterion, epochs):
    model.eval()
    for test_data in test_loader:
        y_pred, embedding = model(test_data.x, test_data.edge_index.long(), test_data.batch)
        loss = criterion(y_pred, test_data.y.unsqueeze(1))
    return loss, embedding        


# __________________________________________________________________________________________
# Load data
file_list = glob.glob('data/processed/data_*')
examples = [torch.load(f) for f in file_list]

# Train/test split
train_size = int(0.7 * len(examples))
test_size = len(examples) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(examples, [train_size, test_size])
print('num training examples: ', len(train_dataset))
print('num testing examples: ', len(test_dataset))

# Set up data loaders
num_graphs_per_batch = 5
train_loader = DataLoader(train_dataset, batch_size=num_graphs_per_batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=num_graphs_per_batch, shuffle=True)

# Load model and set parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_size = 64 # GCNConv size
num_features= 8
model = SimpleGNN().to(device) 
model = model.double()
print(model)
print("Parameters: ", sum(p.numel() for p in model.parameters()))

optimizer = "Adam"
lr = 0.0001
optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=lr)
criterion = nn.MSELoss()
epochs = 30

# Train model
train_losses = []
test_losses = []
for epoch in range(epochs):
    trn_loss, trn_emb = train(train_loader)
    train_losses.append(trn_loss.item())
    tst_loss, tst_emb = test(test_loader)
    test_losses.append(tst_loss.item())
    print(f'Epoch: {epoch}, Train Loss: {trn_loss:.4f}, Test Loss: {tst_loss:.4f}')

# Plot
plot_train_test(epochs, train_losses, test_losses)
