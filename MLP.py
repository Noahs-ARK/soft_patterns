import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layer_dim, num_classes):
        super(MLP, self).__init__()

        self.layer1 = torch.nn.Linear(input_dim, hidden_layer_dim)
        self.layer2 = torch.nn.Linear(hidden_layer_dim, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))
