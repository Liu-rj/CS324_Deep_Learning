from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    def __init__(self, n_inputs, n_hidden, n_classes, lr):
        """
        Initializes multi-layer perceptron object.
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        self.layers = []
        self.layers.append(Linear(n_inputs, n_hidden[0], lr))
        self.layers.append(ReLU())
        for i in range(len(n_hidden) - 1):
            self.layers.append(Linear(n_hidden[i], n_hidden[i + 1], lr))
            self.layers.append(ReLU())
        self.layers.append(Linear(n_hidden[-1], n_classes, lr))
        self.layers.append(SoftMax())

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
