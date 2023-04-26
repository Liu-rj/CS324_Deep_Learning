from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.W_hx = nn.Linear(input_dim, hidden_dim)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim)
        self.W_ph = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        # Implementation here ...
        hidden = nn.Tanh(self.W_hx(x) + self.W_hh(hidden))
        outputs = self.W_ph(hidden)
        return outputs, hidden

    # add more methods here if needed
