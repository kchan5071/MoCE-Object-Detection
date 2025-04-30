import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from data_generator import generate_simple_lstm_data

# Generate sine wave data
t = np.linspace(0, 100, 1000)
data, _ = generate_simple_lstm_data(
        n_samples=12000,
        n_features=3,
        seq_length=60,
        noise_level=0.1,
        random_seed=42
    )

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X, y = create_sequences(data, seq_length)

# Convert data to tensors
trainX = torch.tensor(X[:, :, None], dtype=torch.float32)
trainY = torch.tensor(y[:, None], dtype=torch.float32)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Instantiate the model
input_dim = 1
hidden_dim = 32
layer_dim = 2
output_dim = 1
model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(trainX)
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Make predictions
predicted = model(trainX).detach().numpy()

# Plot the results
plt.plot(t[seq_length:], data[seq_length:], label='Original Data')
plt.plot(t[seq_length:], predicted, label='Predicted Data')
plt.legend()
plt.show()