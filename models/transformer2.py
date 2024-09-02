# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 21:23:35 2024

@author: jorgels
"""

selected = objective_X['names'].isna()
objective_X =  objective_X.loc[~selected]

objective_X['player_id'] = objective_X['names'].astype('category').cat.codes
objective_X = objective_X.drop('names', axis=1)


# Identify categorical columns
categorical_columns = objective_X.select_dtypes(include=['object', 'category']).columns

# Convert each categorical column to numerical representation
for col in categorical_columns:
    objective_X[col] = objective_X[col].astype('category').cat.codes
    
scaler = StandardScaler()
float_columns = objective_X.select_dtypes(include=['float', 'int']).columns

objective_X[float_columns] = scaler.fit_transform(objective_X[float_columns])
    
features = objective_X
target = train_y

grouped = features.groupby('player_id')
sequences = [group.drop(columns=['player_id']).values for _, group in grouped]
targets = target.groupby(objective_X['player_id']).first().values  # Example target for each player
player_ids = [key for key, _ in grouped]


# Check player_ids range
print(f"player_vocab_size: {objective_X['player_id'].nunique()}")
print(f"Max player_id: {np.max(player_ids)}")
print(f"Min player_id: {np.min(player_ids)}")

# Ensure player_ids are within range
assert np.max(player_ids) < objective_X['player_id'].nunique(), "player_id out of range"



import numpy as np
import torch

def pad_sequences(sequences, max_len=None, padding_value=0):
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        seq = np.array(seq, dtype=np.float32)
        padded_seq = np.pad(seq, ((0, max_len - len(seq)), (0, 0)), mode='constant', constant_values=padding_value)
        padded_sequences.append(padded_seq)
    return np.array(padded_sequences, dtype=np.float32)

padded_sequences = pad_sequences(sequences)

# Create mask function
def create_mask(padded_sequences, padding_value=0):
    mask = np.not_equal(padded_sequences, padding_value).all(axis=-1)
    return mask

mask = create_mask(padded_sequences)

# Convert to tensors
padded_sequences = torch.tensor(padded_sequences, dtype=torch.float32)
player_ids = torch.tensor(player_ids, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.float32)
mask = torch.tensor(mask, dtype=torch.bool)

import torch
import torch.nn as nn
import torch.optim as optim

class TransformerModel(nn.Module):
    def __init__(self, nfeatures, nhead, nhid, nlayers, noutput, player_vocab_size, player_embedding_dim, max_seq_len=500):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.player_embedding = nn.Embedding(player_vocab_size, player_embedding_dim)
        self.positional_encoding = self.create_positional_encoding(max_seq_len, nfeatures)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=nfeatures, nhead=nhead, dim_feedforward=nhid)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(nfeatures + player_embedding_dim, noutput)
        
    def create_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: [max_seq_len, 1, d_model]
        return pe
        
    def forward(self, src, player_ids, src_key_padding_mask):
        player_embed = self.player_embedding(player_ids).unsqueeze(1)  # Shape: [batch_size, 1, embedding_dim]
        src = src + self.positional_encoding[:src.size(0), :]
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = torch.cat((output.mean(0), player_embed.squeeze(1)), dim=1)
        output = self.decoder(output)
        return output

# Ensure shapes
n_features = padded_sequences.shape[2]  # 23 features
n_players = objective_X['player_id'].nunique()
player_embedding_dim = 16
n_output = 1

# Define the model, criterion, and optimizer
nhead = 1  # Adjust this depending on your model needs
nhid = 256  # Adjust this based on your neural network settings
nlayers = 4  # Number of Transformer layers

max_seq_len = padded_sequences.shape[1]  # Expected max sequence length

model = TransformerModel(n_features, nhead=nhead, nhid=nhid, nlayers=nlayers,
                         noutput=n_output, player_vocab_size=n_players, player_embedding_dim=player_embedding_dim,
                         max_seq_len=max_seq_len)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if batch sizes and sequence lengths match across padded_sequences and mask
assert padded_sequences.shape[0] == mask.shape[0] == player_ids.shape[0]
assert padded_sequences.shape[1] == mask.shape[1]

# Training loop
num_epochs = 10  # Example number of epochs

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Permute padded_sequences to match the Transformer input requirements: [seq_len, batch_size, features]
    src = padded_sequences.permute(1, 0, 2)  # Shape: [seq_len, batch_size, features]

    # Create the source key padding mask
    src_key_padding_mask = mask  # Ensure this is [batch_size, seq_len]

    # Forward pass
    output = model(src, player_ids, src_key_padding_mask)
    loss = criterion(output.squeeze(), targets)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Optionally save the model
# torch.save(model.state_dict(), 'transformer_model.pth')


