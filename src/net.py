import torch
from torch import nn

class ValueNet(nn.Module):
    def __init__(self, input_size=16*16*4):
        super(ValueNet, self).__init__()
        layers = []
        layers.extend([nn.Linear(input_size, 256), nn.ReLU()])
        layers.extend([nn.Linear(256, 16), nn.ReLU()])
        layers.extend([nn.Linear(16, 1)])
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.layers(x)
        return out