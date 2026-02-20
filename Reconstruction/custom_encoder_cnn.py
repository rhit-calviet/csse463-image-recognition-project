import torch
import torch.nn as nn

class SketchEncoder(nn.Module):

    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        
        # Input: (batch, in_channels, 28, 28)
        self.features = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            # Block 2: 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            # Block 3: 7x7 -> 3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )
        
        # After features, the shape is (batch, 128, 3, 3)
        # 128 * 3 * 3 = 1152
        
        self.latent_layer = nn.Sequential(
            nn.Linear(128 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, latent_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        z = self.latent_layer(x)
        return z