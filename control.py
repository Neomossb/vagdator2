import os.path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn


script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'seattle-weather.csv')

df = pd.read_csv(csv_path)

print(df.head())



# Load data
df['weather_label'] = LabelEncoder().fit_transform(df['weather'])

# Define sequence length (e.g., x days)
SEQ_LEN = 7

# Prepare sequences
def create_flattened_sequences(data, seq_len):
    inputs = []
    labels = []
    for i in range(len(data) - seq_len):
        # Flatten past 'seq_len' days into a single input vector
        seq = data.iloc[i:i + seq_len][['precipitation', 'temp_max', 'temp_min', 'wind']].values.flatten()
        label = data.iloc[i + seq_len]['weather_label']
        inputs.append(seq)
        labels.append(label)
    return np.array(inputs), np.array(labels)

from torch.utils.data import Dataset

class WeatherDataset(Dataset):
    def __init__(self, inputs, labels):
        """
        Initialize the dataset with inputs and labels.
        :param inputs: Feature data (e.g., flattened sequences as numpy arrays or tensors).
        :param labels: Corresponding labels for the data.
        """
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieve a single sample and its label by index.
        :param idx: Index of the sample.
        :return: Tuple (input, label).
        """
        return self.inputs[idx], self.labels[idx]

sequences, labels = create_flattened_sequences(df, SEQ_LEN)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)



input_size = SEQ_LEN * 4  # 4 features (precipitation, temp_max, temp_min, wind) per day

class WeatherClassifier(nn.Module):
    def __init__(self):
        super(WeatherClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First layer
        self.fc2 = nn.Linear(128, 64)        # Second layer
        self.fc3 = nn.Linear(64, 5)          # Output layer (5 classes for weather)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

train_dataset = WeatherDataset(X_train, y_train)
test_dataset = WeatherDataset(X_test, y_test)







class WeatherPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(WeatherPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # Use final hidden state
        out = self.fc(h_n[-1])  # Pass through fully connected layer
        return out

model = WeatherClassifier()



import torch.optim as optim

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.85)

# Create DataLoaders for training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training and validation loop
for epoch in range(50):  # Train for 50 epochs
    # Training loop
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Step the learning rate scheduler
    scheduler.step()

    # Validation loop
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print epoch statistics
    print(f"Epoch {epoch + 1}/50, Train Loss: {train_loss / len(train_loader):.4f}, "
          f"Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {100 * correct / total:.2f}%")

