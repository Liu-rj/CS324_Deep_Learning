import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_inputs, n_hidden[0]))
        for i in range(len(n_hidden) - 1):
            self.layers.append(nn.Linear(n_hidden[i], n_hidden[i + 1]))
        self.layers.append(nn.Linear(n_hidden[-1], n_classes))
        self.n_classes = n_classes

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        out = x
        for it, layer in enumerate(self.layers):
            out = layer(out)
            if it != len(self.layers) - 1:
                out = F.relu(out)
            else:
                out = (
                    torch.sigmoid(out) if self.n_classes == 2 else F.softmax(out, dim=0)
                )
        return out
