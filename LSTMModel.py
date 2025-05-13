import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import ModelCollector
import LSTMDataLoader





# Define the LSTM model
class LSTMTrainingModel(nn.Module):
    def __init__(self, input_models, hidden_dim, layer_dim, output_dim=5, max_index=100):
        super(LSTMTrainingModel, self).__init__()
        self.input_dim = input_models # 5 dimensional predictions from each model
        self.outputs_per_model = 5
        self.number_of_models = 3
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.sequence_length = 2
        self.batch_size = 2
        
        self.lstm = nn.LSTM(self.input_dim, hidden_size=15, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        #data loading
        self.current_index = 0
        self.split = [.7, .2, .1] #train, val, test
        self.max_index = max_index

        #initialize the models
        self.yolo_model = ModelCollector.YoloModel('Yolov5/best.pt')
        self.resnet_model = ModelCollector.ResNetModel('ResNet/best.pth')
        self.detr_model = ModelCollector.WillFlow()
        self.detr_model.load_state_dict(
            torch.load("VisionTransformer/best.pth", map_location=torch.device('cpu'), weights_only=True)
        )

        self.train_sample_indexes = []
        self.val_sample_indexes = []
        self.test_sample_indexes = []

        self.create_train_test_val_indexes(max_index)

        # self.train_batches = self.break_into_batches(self.train_sample_indexes)
        # self.val_batches = self.break_into_batches(self.val_sample_indexes)
        # self.test_batches = self.break_into_batches(self.test_sample_indexes)

    def create_train_test_val_indexes(self, num_samples):
        """
        Create batch indexes for the LSTM model.

        Args:
            num_samples: Number of samples in the dataset.
            sequence_length: Length of each sequence.

        Returns:
            batch_indexes: List of batch indexes.
        """
        batch_indeces = list(range(self.max_index + 1))
        # Shuffle the batch indexes
        np.random.shuffle(batch_indeces)
        # Split the batch indexes into train, val, and test sets
        for i in range(num_samples):
            if i < num_samples * self.split[0]:
                self.train_sample_indexes.append(batch_indeces[i])
            elif i < num_samples * (self.split[0] + self.split[1]):
                self.val_sample_indexes.append(batch_indeces[i])
            else:
                self.test_sample_indexes.append(batch_indeces[i])
    

    def break_into_batches(self, instance):
        """
        Break the data into batches.

        Args:
            batch_size: Size of each batch.
            instance: List of predicionts or ground truth labels. (3d array)

        Returns:
            batches: List of batches. (3d nparray in shape (batch_size, sequence_length, input_dim))
        """
        # transpose the instance to have shape (sequence_length, batch_size, input_dim)
        #if 2d array, add a dimension
        if len(instance.shape) == 2:
            instance = np.expand_dims(instance, axis=0)
        instance = np.transpose(instance, (2, 0, 1))
        

        # Calculate the number of batches
        num_batches = len(instance) // self.batch_size
        # Create batches
        batches = []
        for i in range(num_batches):
            batch = instance[i * self.batch_size:(i + 1) * self.batch_size]
            batches.append(batch)
        # Convert to numpy array
        batches = np.array(batches)
        return batches.squeeze()
    

    def get_images_and_labels(self, batch_size):
        """
        Get images and labels for the LSTM model.

        The expert predictions and ground truth labels are the training data.
        """
        labels = np.zeros((self.outputs_per_model, self.max_index))
        expert_predictions = np.zeros((self.number_of_models, self.outputs_per_model, self.max_index), dtype=np.float32)
        for i in range(self.max_index):
            image_path, _, image, labels[:,i] = LSTMDataLoader.get_image_and_label(i)
            expert_predictions[:,:,i] = ModelCollector.inference_on_models(
                self.yolo_model,
                self.resnet_model,
                self.detr_model,
                image,
                image_path
            )

        return expert_predictions, labels

    def forward(self, x):
        """
        Forward pass for the LSTM model.

        Args:
            x: Tensor of shape (batch_size, sequence_length, input_dim) containing input data.

        Returns:
            output: Tensor of shape (batch_size, output_dim).
        """   

        # Pass the input through the LSTM layer
        lstm_out, _ = self.lstm(x)
        # Get the last time step output
        lstm_out = lstm_out[-1, :, :]
        # Pass the output through the fully connected layer
        out = self.fc(lstm_out)

        return out

    def training_loop(self, train_data, train_labels, num_epochs=100, learning_rate=0.001):
        """
        Train the LSTM model.
        Args:
            train_data: Tensor of shape (num_samples, sequence_length, input_dim) containing training data.
            train_labels: Tensor of shape (num_samples, output_dim) containing training labels.
            num_epochs: Number of epochs to train the model.
            learning_rate: Learning rate for the optimizer.
        """
        # Ensure train_data and train_labels are torch.Tensors
        if isinstance(train_data, np.ndarray):
            train_data = torch.tensor(train_data, dtype=torch.float32)
        if isinstance(train_labels, np.ndarray):
            train_labels = torch.tensor(train_labels, dtype=torch.float32)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        losses = []

        # Training loop
        for epoch in range(num_epochs):
            # Forward pass
            output = self(train_data)
            # Compute loss
            loss = criterion(output, train_labels)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Print loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

            if (epoch + 1) % 50 == 0:
                # Save the model checkpoint
                torch.save(self.state_dict(), f'lstm_model_epoch_{epoch + 1}.pth')
                print(f'Model saved at epoch {epoch + 1}')

        return losses
    
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
    

def main():
    # Example usage
    input_dim = 15
    hidden_dim = 15
    layer_dim = 3
    max_index = 50

    images_to_load = 10

    lstm_model = LSTMTrainingModel(input_dim, hidden_dim, layer_dim, max_index=max_index)
    lstm_model.create_train_test_val_indexes(max_index)
    train_data, train_labels = lstm_model.get_images_and_labels(images_to_load)

    print("Training data shape:", train_data.shape)
    print("Training labels shape:", train_labels.shape)

    train_data = train_data.reshape(15, lstm_model.max_index)
    print("Reshaped Training data shape:", train_data.shape)

    print(train_data)

    # Convert to PyTorch tensors
    train_labels_batches = lstm_model.break_into_batches(train_labels)

    # Reshape training data to (batch_size, sequence_length, input_dim)
    train_data_batches= lstm_model.break_into_batches(train_data)
    #reshape to (batch_size, sequence_length, input_dim)

    # Ensure batches are torch.Tensors
    train_labels_batches = torch.tensor(train_labels_batches, dtype=torch.float32)
    train_data_batches = torch.tensor(train_data_batches, dtype=torch.float32)

    print("Batched training data shape:", train_data_batches.shape, type(train_data_batches))
    print("Batched training labels shape:", train_labels_batches.shape, type(train_labels_batches))

    print("FINISHED DATA LOADING")
    # Train the model
    losses = lstm_model.training_loop(train_data=train_data_batches, train_labels=train_labels_batches, num_epochs=200, learning_rate=0.001)

    # Predict and print the output
    test = np.zeros((2, 1, 15))
    #convert to torch tensor
    test = torch.tensor(test, dtype=torch.float32)
    print("Test shape:", test.shape)
    output = lstm_model.forward(test)
    print("Output shape:", output.shape)

    # Plot the loss
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')


if __name__ == "__main__":
    main()