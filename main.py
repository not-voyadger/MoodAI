#--------------------------------
# Full definition of MoodAI - Convolutional Neural Network for emotion recognition via webcam, model definition, training and testing in this single file.
# Arsenii Fadieiev 06.11.2025
#--------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

DATA_DIR = "data" #Path to the dataset folder. Should contain 'train' and 'test' subfolders with images organized in class subfolders.
BATCH_SIZE = 32 # Number of images processed in one iteration. Smaller batch sizes (e.g., 16) use less memory and may generalize better, but training will be slower.
EPOCHS = 10 # Number of full passes through the training dataset.
LEARNING_RATE = 0.001 # Step size for the optimizer. A higher value can make training faster but unstable.
MODEL_PATH = "face_model.pth" # File path to save or load the trained model.
IMAGE_SIZE = 48 #Size (height and width) to which all images will be resized. Should match the input size expected by the network.       

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# This section transforms image to needed format - image becomes black and white, resizes to the size of 48 x 48 pixels, converts to PyTorch-tensor format and normalizes it.
transform = transforms.Compose([
    transforms.Grayscale(),      
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_dataset.classes)
print(f"Found {num_classes} classes:", train_dataset.classes)

# Model implementation
class MoodAI(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Сonvolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2)
        )
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * (IMAGE_SIZE//8) * (IMAGE_SIZE//8), 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

model = MoodAI(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# This section trains created model by going through images EPOCH number of times.
for epoch in range(EPOCHS):
    model.train()
    current_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # Reset gradients to zero.
        outputs = model(images)
        loss = criterion(outputs, labels) # Compute the loss.
        loss.backward() # Backpropagation.
        optimizer.step()

        current_loss += loss.item()

    avg_loss = current_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# Model testing to evaluate guessing accuracy.
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
