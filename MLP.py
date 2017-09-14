import torch

class MLP:
    def __init__(self, input_dim, hidden_layer_dim, num_classes):
        self.layer1 = nn.Linear(input_dim, hidden_layer_dim)
        self.layer2 = nn.Linear(hidden_layer_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))
