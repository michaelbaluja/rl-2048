import torch
from torch import nn
import random
from collections import namedtuple

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
    

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)