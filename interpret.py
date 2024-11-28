import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import os
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

img_width = 370
img_height = 330

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data/data3")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
    transforms.Resize((img_width, img_height)),               # Ensure size is consistent
    transforms.ToTensor(),                        # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(root=data_path, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

"""
class WeatherClassifier(nn.Module):
    def __init__(self):
        super(WeatherClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Dropout layers
        self.dropout_conv = nn.Dropout2d(0.25)  # Dropout for convolutional layers
        self.dropout_fc = nn.Dropout(0.5)  # Dropout for fully connected layers

        # Fully connected layers
        self.fc1 = nn.Linear(64 * (357 // 8) * (306 // 8), 128)  # Adjust for downsampling
        self.fc2 = nn.Linear(128, 5)  # 5 output classes

    def forward(self, x):
        # Convolutional layers with dropout
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = self.dropout_conv(x)  # Apply dropout after first conv layer

        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # x = self.dropout_conv(x)  # Apply dropout after second conv layer

        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        # x = self.dropout_conv(x)  # Apply dropout after third conv layer

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        # x = self.dropout_fc(x)  # Apply dropout after first fully connected layer
        x = self.fc2(x)  # Output layer (no activation here, handled by loss function)
        return x
"""


class WeatherClassifier(nn.Module):
    def __init__(self):
        super(WeatherClassifier, self).__init__()
        # Flatten the input image (img_width * img_height)
        input_size = img_width * img_height

        # Define the three linear layers
        self.fc1 = nn.Linear(input_size, 128)  # First layer
        self.fc2 = nn.Linear(128, 64)  # Second layer
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        # Flatten the input image to a vector
        x = x.view(x.size(0), -1)  # Reshape to (batch_size, input_size)

        # Pass through the linear layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# Compute class weights
# class_weights = compute_class_weight('balanced', classes=np.arange(5), y=dataset.targets)
# class_weights = torch.tensor(class_weights, dtype=torch.float32)



model = WeatherClassifier()
# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.85)

# Training loop
for epoch in range(50):  # Train for 50 epochs
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation loop
    scheduler.step()

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch + 1}/{50}, Train Loss: {train_loss / len(train_loader):.4f}, "
          f"Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {100 * correct / total:.2f}%")

