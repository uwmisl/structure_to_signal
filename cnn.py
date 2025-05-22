import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
import re
import os

def one_hot_encode(strings, bases):
    char_to_index = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'B': 4, 'J': 5, 'K': 6, 'P': 7, 'Sc': 8, 'V': 9, 'X': 10, 'Zn': 11, 'Sn': 12, 'Za': 13}
    chars = ['A', 'T', 'G', 'C', 'B', 'J', 'K', 'P', 'Sc', 'V', 'X', 'Zn', 'Sn', 'Za']
    num_chars = len(chars)
    encoded_strings = []
    
    for string, base in zip(strings, bases):
        one_hot_array = np.zeros((len(string), num_chars))        
        for i, char in enumerate(string):
            if char == 'S' and base == 'Sc':
                char_index = char_to_index['Sc']
            elif char == 'S' and base == 'Sn':
                char_index = char_to_index['Sn']
            elif char == 'Z' and base == 'Za':
                char_index = char_to_index['Za']
            elif char == 'Z' and base == 'Zn':
                char_index = char_to_index['Zn']
            else:
                char_index = char_to_index[char]

            one_hot_array[i, char_index] = 1
        
        encoded_strings.append(one_hot_array)
    
    return encoded_strings

def r_squared(true_values, predicted_values):
    true_values = np.array(true_values)  
    predicted_values = np.array(predicted_values)  
    res = stats.mstats.linregress(true_values, predicted_values)
    r_squared = res.rvalue**2
    return r_squared

class CNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8 * 2, 16)
        self.fc2 = nn.Linear(16, output_size)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 8 * 2)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(train_loader, epochs, model, optimizer, criterion):
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.permute(0, 2, 1))  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss}")

def test(test_loader, model, criterion):
    test_loss = 0.0
    predicted_values = []
    true_values = []
    kmers = []
    model.eval()
    with torch.no_grad():
        for inputs, labels, kmer in test_loader:  
            outputs = model(inputs.permute(0, 2, 1))  
            loss = criterion(outputs, labels)
            true_values.extend(labels.numpy())
            predicted_values.extend(outputs.numpy())
            kmers.extend(kmer)

            test_loss += loss.item()
        print(f"Test Loss: {test_loss / len(test_loader)}")
    
    rmse = np.sqrt(mean_squared_error(np.array(true_values), np.array(predicted_values)))

    return true_values, predicted_values, kmers, rmse

class XNADataset(Dataset):
    def __init__(self, inputs, labels, kxmrs):
        self.inputs = inputs
        self.labels = labels
        self.kxmrs = kxmrs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.kxmrs[idx]

def run_xna_basedrop(
    dfs_dict,
    num_runs=1,
    epochs=100,
    batch_size=128,
    use_ATGC=True,
    output_dir="cnn_xna_basedrop_predictions"
):
    '''
    Runs a base dropout experiment for each unique XNA base (trained on all other XNA datasets plus canonical dataset)
    Outputs a kmer table wih model predicted values for each kmer
    '''
    os.makedirs(output_dir, exist_ok=True)  

    base_names = list(dfs_dict.keys())
    df_ATGC = dfs_dict['ATGC'] if use_ATGC else None

    for holdout_base in base_names:
        if holdout_base == 'ATGC':
            continue  # Only test on XNAs

        print(f"HOLDING OUT: {holdout_base}")
        for run in range(num_runs):
            print(f"Run {run + 1}")

            # Create training set (excluding the holdout)
            train_dfs = [df for base, df in dfs_dict.items() if base != holdout_base and base != 'ATGC']
            if use_ATGC:
                train_dfs.append(df_ATGC)

            train_df = pd.concat(train_dfs, ignore_index=True)
            test_df = dfs_dict[holdout_base]

            # Encoding
            train_df['one_hot_encoded'] = one_hot_encode(train_df['KXmer'], train_df['base'])
            test_df['one_hot_encoded'] = one_hot_encode(test_df['KXmer'], test_df['base'])

            # Convert to tensors
            X_train = torch.tensor(train_df['one_hot_encoded'].tolist(), dtype=torch.float32)
            y_train = torch.tensor(train_df['Mean level'].values.reshape(-1, 1), dtype=torch.float32)
            X_test = torch.tensor(test_df['one_hot_encoded'].tolist(), dtype=torch.float32)
            y_test = torch.tensor(test_df['Mean level'].values.reshape(-1, 1), dtype=torch.float32)

            train_dataset = XNADataset(X_train, y_train, train_df['KXmer'].tolist())
            test_dataset = XNADataset(X_test, y_test, test_df['KXmer'].tolist())

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Initialize model
            model = CNN(input_channels=14, output_size=1)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Train model
            train(train_loader, epochs, model, optimizer, criterion)

            # Evaluate model
            true, pred, kmer, rmse = test(test_loader, model, criterion)
            r2 = r_squared(true, pred)

            # Save predictions
            predictions_df = pd.DataFrame({
                'kmer': kmer,
                'prediction': pred,
                'true': true
            })

            filename = f"predictions_holdout-{holdout_base}_run-{run+1}.csv"
            filepath = os.path.join(output_dir, filename)
            predictions_df.to_csv(filepath, index=False)

            print(f"Saved predictions to: {filepath}")
            print(f"r²: {r2:.4f}, RMSE: {rmse:.4f}")
        print("-" * 40)

# Main
# _________________________________________
base_paths = {
    'ATGC': 'data/ATGC_r9.4.1.csv',
    'B': 'data/B_r9.4.1.csv',
    'J': 'data/J_r9.4.1.csv',
    'K': 'data/Kn_r9.4.1.csv',
    'P': 'data/P_r9.4.1.csv',
    'Sc': 'data/Sc_r9.4.1.csv',
    'V': 'data/V_r9.4.1.csv',
    'X': 'data/Xt_r9.4.1.csv',
    'Zn': 'data/Zn_r9.4.1.csv',
    'Za': 'data/Za_r9.4.1.csv',
    'Sn': 'data/Sn/Sn_r9.4.1.csv'
}

dfs = {}
for base, path in base_paths.items():
    df = pd.read_csv(path)
    df['base'] = base
    dfs[base] = df

run_xna_basedrop(dfs_dict=dfs, num_runs=1, epochs=100, batch_size=128, use_ATGC=True)