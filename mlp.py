from torch.nn import Linear, Module
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

        # This code is a bit strange in order to support previous versions: if num_layers is 2,
        # create two member layers (layer1 and layer2). Otherwise, create a list of layers of size num_layers
        if num_layers == 1:
            self.layers = [Linear(input_dim, num_classes)]
        elif num_layers == 2:
            self.layer1 = Linear(input_dim, hidden_layer_dim)
            self.layer2 = Linear(hidden_layer_dim, num_classes)
        else:
            self.layers = [Linear(input_dim, hidden_layer_dim)]

            for i in range(1, num_layers-1):
                self.layers.append(Linear(hidden_layer_dim, hidden_layer_dim))

            self.layers.append(Linear(hidden_layer_dim, num_classes))


    def forward(self, x):
        if self.num_layers == 2:
            return self.layer2(relu(self.layer1(x)))
        else:
            res = self.layers[0](x)

            for i in range(len(self.layers)-1):
                res = self.layers[i].relu(res)
            return res
