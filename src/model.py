import torch
import torch.nn as nn
import numpy as np


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.activation(x)
        x = self.bn(x)
        return x


class CNN(nn.Module):
    def __init__(self, image_sizes, nb_classes):
        super().__init__()
        self.model = nn.Sequential(
            Conv(3, 32, 3, 2, 1),
            Conv(32, 64, 3, 2, 1),
            Conv(64, 128, 3, 2, 1),
            Conv(128, 256, 3, 2, 1),
            Conv(256, 128, 3, 2, 1),
            Conv(128, 64, 3, 2, 1),
            Conv(64, 32, 3, 2, 1))

        fe_output = np.prod(self.model(torch.zeros(1, 3, *image_sizes, device="cpu")).shape[1:])

        self.dense = nn.Linear(fe_output, nb_classes)

    def forward(self, inputs):
        x = self.model(inputs)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        return x
