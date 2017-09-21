from torch.nn import Linear, Module
from torch.nn.functional import relu


class MLP(Module):
    """ A multilayer perceptron with one hidden ReLU layer """
    def __init__(self,
                 input_dim,
                 hidden_layer_dim,
                 num_classes):
        super(MLP, self).__init__()

        self.layer1 = Linear(input_dim, hidden_layer_dim)
        self.layer2 = Linear(hidden_layer_dim, num_classes)

    def forward(self, x):
        return self.layer2(relu(self.layer1(x)))
