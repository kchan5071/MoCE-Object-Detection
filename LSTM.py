import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from data_generator import generate_simple_lstm_data




# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def load_data(self):
        pass

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    # Define the training function
    def train_model(model, criterion, optimizer, train_loader, num_epochs):
        for epoch in range(num_epochs):
            for i, (inputs, labels) in enumerate(train_loader):
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    