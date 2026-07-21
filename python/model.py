import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_m = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)

        out = out + x_m

        return out
        

class Res8(nn.Module):
    def __init__(self, n_channels=45, num_classes=15, n_blocks=3):
        super().__init__()
        self.input_conv = nn.Conv2d(1, n_channels, kernel_size=3, padding=1)

        self.blocks = nn.Sequential(
            *[ResidualBlock(n_channels) for _ in range(n_blocks)]
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_channels, num_classes)

    def forward(self, x):
        x = self.input_conv(x)

        x = self.blocks(x)

        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x