from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################


class LSTM(nn.Module):
    def __init__(self, seq_length, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.W_fx = nn.Linear(input_dim, hidden_dim)
        self.W_ix = nn.Linear(input_dim, hidden_dim)
        self.W_gx = nn.Linear(input_dim, hidden_dim)
        self.W_ox = nn.Linear(input_dim, hidden_dim)

        self.W_fh = nn.Linear(hidden_dim, hidden_dim)
        self.W_ih = nn.Linear(hidden_dim, hidden_dim)
        self.W_gh = nn.Linear(hidden_dim, hidden_dim)
        self.W_oh = nn.Linear(hidden_dim, hidden_dim)

        self.W_ph = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Implementation here ...
        h = torch.zeros(self.hidden_dim, device="mps")
        c = torch.zeros(self.hidden_dim, device="mps")
        for i in range(self.seq_length):
            xi = x[:, i].unsqueeze(-1)
            g = torch.tanh(self.W_gx(xi) + self.W_gh(h))
            i = torch.sigmoid(self.W_ix(xi) + self.W_ih(h))
            f = torch.sigmoid(self.W_fx(xi) + self.W_fh(h))
            o = torch.sigmoid(self.W_ox(xi) + self.W_oh(h))
            c = g * i + c * f
            h = torch.tanh(c) * o
        p = self.W_ph(h)
        return p

    # add more methods here if needed
