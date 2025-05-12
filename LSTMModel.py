import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from data_generator import generate_simple_lstm_data




# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, image):
        # Pass the image through the LSTM
        lstm_out, _ = self.lstm(image)
        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]
        # Pass through the fully connected layer
        output = self.fc(last_out)
        return output
    def train(self, train_data, train_labels, num_epochs=100, learning_rate=0.001):
        """
        Train the LSTM model.
        Args:
            train_data: Tensor of shape (num_samples, sequence_length, input_dim) containing training data.
            train_labels: Tensor of shape (num_samples, output_dim) containing training labels.
            num_epochs: Number of epochs to train the model.
            learning_rate: Learning rate for the optimizer.
        """
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(train_data)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        return self
    
    def predict(self, input_data):
        """
        Predict the output for the given input data.
        Args:
            input_data: Tensor of shape (batch_size, sequence_length, input_dim) containing input data.
        Returns:
            output: Tensor of shape (batch_size, output_dim) containing the predicted output.
        """
        self.eval()
        with torch.no_grad():
            output = self(input_data)
        return output

