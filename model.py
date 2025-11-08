#--------------------------------
# Model definition of MoodAI - Convolutional Neural Network for emotion recognition via webcam.
# Arsenii Fadieiev 06.11.2025
#--------------------------------

import torch
import torch.nn as nn

IMAGE_SIZE = 48

def Conv3x3ReLU(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True)
    )

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            Conv3x3ReLU(channels, channels),
            Conv3x3ReLU(channels, channels)
        )
        
    def forward(self, x):
        return x + self.block(x)
    

class MoodAI(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            ResidualBlock(32),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResidualBlock(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResidualBlock(128),
            nn.MaxPool2d(2, 2)
        )

        self.dropout = nn.Dropout2d(0.3)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x