from torch import nn
from torch.nn import Linear, Module,ModuleList
from torch.nn.functional import relu


class MLP(Module):
    """ A multilayer perceptron with one hidden ReLU layer """
    def __init__(self,
                 input_dim,
                 hidden_layer_dim,
                 num_layers,
                 num_classes,
                 dropout=0,
                 legacy=1):
        super(MLP, self).__init__()

        self.num_layers = num_layers
        self.legacy = legacy

        if dropout:
            dropout_layer = nn.Dropout(p=dropout)
        # This code is a bit strange in order to support previous versions: if num_layers is 2,
        # create two member layers (layer1 and layer2). Otherwise, create a list of layers of size num_layers
        layers = []

        for i in range(num_layers):
            d1 = input_dim if i == 0 else hidden_layer_dim
            d2 = hidden_layer_dim if i < (num_layers - 1) else num_classes

            layer = Linear(d1, d2)

            # if dropout:
            #     layer = dropout_layer(layer)

            layers.append(layer)

        if legacy and num_layers == 2:
            self.layer1 = layers[0]
            self.layer2 = layers[1]
        else:
            self.layers = ModuleList(layers)



    def forward(self, x):
        if self.legacy and self.num_layers == 2:
            return self.layer2(relu(self.layer1(x)))
        else:
            res = self.layers[0](x)

            for i in range(len(self.layers)-1):
                res = self.layers[i].relu(res)
            return res
