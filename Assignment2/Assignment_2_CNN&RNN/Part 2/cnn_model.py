import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        """
        Initializes CNN object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        """
        super(CNN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.layers = []
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(n_channels, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        )
        self.layers.append(nn.MaxPool2d(3, stride=2, padding=1))
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        )
        self.layers.append(nn.MaxPool2d(3, stride=2, padding=1))
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )
        self.layers.append(nn.MaxPool2d(3, stride=2, padding=1))
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(256, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        )
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        )
        self.layers.append(nn.MaxPool2d(3, stride=2, padding=1))
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        )
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        )
        self.layers.append(nn.MaxPool2d(3, stride=2, padding=1))
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        """
        Performs forward pass of the input.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        h = x
        for layer in self.layers:
            h = layer(h)
        h = self.fc(h.view(-1, 512))
        out = F.softmax(h, dim=0)
        return out
