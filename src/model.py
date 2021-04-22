import torch
import torch.nn as nn
import numpy as np

class CNN(nn.Module):
    def __init__(self, image_sizes, nb_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 2, 1),
            nn.ReLU())
        
        fe_output = np.prod(self.model(torch.zeros(1, 3, *image_sizes, device="cpu")).shape[1:])

        self.dense = nn.Linear(fe_output, nb_classes)

    def forward(self, inputs):
        x = self.model(inputs)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        return x