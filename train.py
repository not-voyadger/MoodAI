#--------------------------------
# Model training.
# Arsenii Fadieiev 06.11.2025
#--------------------------------

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

from model import MoodAI, IMAGE_SIZE

DATA_DIR = "data"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_PATH = "face_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(train_dataset.classes)
print(f"Found {num_classes} classes:", train_dataset.classes)

model = MoodAI(num_classes).to(device)

if os.path.exists(MODEL_PATH):
    print(f"Loading existing model weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print("No saved model found — training from scratch.")
          
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model finished training and all additions are saved to {MODEL_PATH}")