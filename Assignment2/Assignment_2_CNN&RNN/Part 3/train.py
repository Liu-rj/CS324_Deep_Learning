import argparse
import time
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN


def train(config, device):
    # Initialize the model that we are going to use
    model = VanillaRNN(
        config.input_length,
        config.input_dim,
        config.num_hidden,
        config.num_classes,
        config.batch_size,
    ).to(device)

    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    steps, acces, losses = [], [], []
    for step, (inputs, targets) in enumerate(data_loader):
        # Add more code here ...
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = torch.zeros(1, config.num_hidden, device="cuda")

        for i in range(config.input_length):
            pred, hidden = model(inputs[:, i].unsqueeze(-1), hidden)

        optimizer.zero_grad()
        loss = criterion(pred, targets)
        loss.backward()
        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optimizer.step()

        # Add more code here ...

        loss = loss.item()
        accuracy = (torch.argmax(pred, dim=1) == targets).sum().item() / pred.shape[0]

        if step % 10 == 0:
            steps.append(step)
            losses.append(loss)
            acces.append(accuracy)
            print("Step: {} Loss: {:.4f} Accuracy: {:.4f}".format(step, loss, accuracy))
            # print acuracy/loss here

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print("Done training.")
    return steps, acces, losses


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument(
        "--input_length", type=int, default=5, help="Length of an input sequence"
    )
    parser.add_argument(
        "--input_dim", type=int, default=1, help="Dimensionality of input sequence"
    )
    parser.add_argument(
        "--num_classes", type=int, default=10, help="Dimensionality of output sequence"
    )
    parser.add_argument(
        "--num_hidden",
        type=int,
        default=128,
        help="Number of hidden units in the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of examples to process in a batch",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--train_steps", type=int, default=10000, help="Number of training steps"
    )
    parser.add_argument("--max_norm", type=float, default=10.0)

    config = parser.parse_args()
    # Train the model
    train(config, "cuda")
