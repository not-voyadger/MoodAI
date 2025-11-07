#--------------------------------
# Testing of trained model.
# Arsenii Fadieiev 06.11.2025
#--------------------------------

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys

from model import MoodAI, IMAGE_SIZE

DATA_DIR = "data"
BATCH_SIZE = 32
MODEL_PATH = "face_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(test_dataset.classes)
model = MoodAI(num_classes).to(device)

if os.path.exists(MODEL_PATH):
    print(f"Starting to load existing weights...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print(f"Could not find trained model, please run train.py first.")
    sys.exit(1)

model.eval()

correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")