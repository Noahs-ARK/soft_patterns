from torch.nn import Linear, Module, ModuleList
from torch.nn.functional import relu


class MLP(Module):
    """ A multilayer perceptron with one hidden ReLU layer """
    def __init__(self,
                 input_dim,
                 hidden_layer_dim,
                 num_layers,
                 num_classes):
        super(MLP, self).__init__()

        self.num_layers = num_layers

        # create a list of layers of size num_layers
        layers = []
        for i in range(num_layers):
            d1 = input_dim if i == 0 else hidden_layer_dim
            d2 = hidden_layer_dim if i < (num_layers - 1) else num_classes
            layer = Linear(d1, d2)
            layers.append(layer)

        self.layers = ModuleList(layers)

    def forward(self, x):
        res = self.layers[0](x)
        for i in range(1, len(self.layers)):
            res = self.layers[i](relu(res))
        return res
