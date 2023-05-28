from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from dataset import PalindromeDataset
from lstm import LSTM
from vanilla_rnn import VanillaRNN
from utils import AverageMeter, accuracy


def train(model, data_loader, optimizer, criterion, device, config):
    # TODO set model to train mode
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    model.train()
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here ...
        inputs, targets = batch_inputs.to(device), batch_targets.to(device)
        pred = model(inputs)

        optimizer.zero_grad()
        loss = criterion(pred, targets)
        loss.backward()

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optimizer.step()

        # Add more code here ...
        loss = loss.item()
        acc = accuracy(pred, targets)
        losses.update(loss)
        accuracies.update(acc)
        if step % 10 == 0:
            print(f"[{step}/{len(data_loader)}]", losses, accuracies)
    return losses.avg, accuracies.avg


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, config):
    # TODO set model to evaluation mode
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    model.eval()
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here ...
        inputs, targets = batch_inputs.to(device), batch_targets.to(device)
        pred = model(inputs)

        loss = criterion(pred, targets).item()
        acc = accuracy(pred, targets)
        losses.update(loss)
        accuracies.update(acc)
        if step % 10 == 0:
            print(f"[{step}/{len(data_loader)}]", losses, accuracies)
    return losses.avg, accuracies.avg


def main(config, device, model_name):
    # Initialize the model that we are going to use
    if model_name == "LSTM":
        model = LSTM(
            config.input_length, config.input_dim, config.num_hidden, config.num_classes
        )
    elif model_name == "VanillaRNN":
        model = VanillaRNN(
            config.input_length, config.input_dim, config.num_hidden, config.num_classes
        )
    model.to(device)

    # Initialize the dataset and data loader
    dataset = PalindromeDataset(config.input_length + 1, 2048)
    # Split dataset into train and validation sets
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        dataset, lengths=[0.8, 0.2], generator=generator
    )
    # Create data loaders for training and validation
    train_dloader = DataLoader(train_dataset, config.batch_size)
    val_dloader = DataLoader(val_dataset, config.batch_size)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    train_acces, val_acces = [], []
    train_losses, val_losses = [], []
    for epoch in range(config.max_epoch):
        # Train the model for one epoch
        train_loss, train_acc = train(
            model, train_dloader, optimizer, criterion, device, config
        )

        # Evaluate the trained model on the validation set
        val_loss, val_acc = evaluate(model, val_dloader, criterion, device, config)

        train_acces.append(train_acc)
        val_acces.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    print("Done training.")
    return train_acces, train_losses, val_acces, val_losses


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument(
        "--input_length", type=int, default=19, help="Length of an input sequence"
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
        "--max_epoch", type=int, default=100, help="Number of epochs to run for"
    )
    parser.add_argument("--max_norm", type=float, default=10.0)
    parser.add_argument(
        "--data_size", type=int, default=1000000, help="Size of the total dataset"
    )
    parser.add_argument(
        "--portion_train",
        type=float,
        default=0.8,
        help="Portion of the total dataset used for training",
    )

    config = parser.parse_args()
    # Train the model
    main(config, "cuda", "LSTM")
