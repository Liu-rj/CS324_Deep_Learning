from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    def __init__(self, seq_length, input_dim, hidden_dim, output_dim):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        self.W_hx = nn.Linear(input_dim, hidden_dim)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim)
        self.W_ph = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Implementation here ...
        h = torch.zeros(1, self.hidden_dim, device="cuda")
        for i in range(self.seq_length):
            xi = x[:, i].unsqueeze(-1)
            h = torch.tanh(self.W_hx(xi) + self.W_hh(h))
        outputs = self.W_ph(h)
        return outputs

    # add more methods here if needed
