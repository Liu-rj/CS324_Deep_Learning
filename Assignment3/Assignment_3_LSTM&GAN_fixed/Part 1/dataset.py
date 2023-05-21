from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch.utils.data as data


class PalindromeDataset(data.Dataset):

    def __init__(self, seq_length, total_len, one_hot=False):
        self.seq_length = seq_length
        self.one_hot = one_hot
        self.half_length = math.ceil(seq_length/2)
        self.total_len = total_len
        if self.total_len > 10 ** self.half_length:
            print("Warning: total_len is larger than the maximum possible length. ")
            print("Setting total_len to the maximum possible length. ")
            self.total_len = 10 ** self.half_length
        allowed_max_len = np.iinfo(np.uint64).max
        if self.total_len > allowed_max_len:
            print("Warning: total_len is too large. ")
            print("Setting total_len to the maximum allowed length. ")
            self.total_len = allowed_max_len
        self.mapping = np.eye(10) if one_hot else np.arange(10).reshape(10, 1)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Keep last digit as target label.
        full_palindrome = self.generate_palindrome(idx)
        # Split palindrome into inputs (N-1 digits) and target (1 digit)
        inputs, labels = full_palindrome[0:-1], int(full_palindrome[-1])
        inputs = self.mapping[inputs].astype(np.float32)
        return inputs, labels

    def generate_palindrome(self, idx):
        idx = tuple(map(int, str(idx)))
        left = np.zeros(self.half_length).astype(np.uint64)
        left[-len(idx):] = idx
        if self.seq_length % 2 == 0:
            right = left[::-1]
        else:
            right = left[-2::-1]
        return np.concatenate((left, right))
