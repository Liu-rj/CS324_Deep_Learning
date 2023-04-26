import argparse
import numpy as np
from pytorch_mlp import MLP
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = "20"
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 100
EVAL_FREQ_DEFAULT = 1

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    return (torch.argmax(predictions, dim=1) == targets).sum() / predictions.shape[0]


def train(FLAGS, train_loader, test_loader, device):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    n_hidden = [int(num) for num in FLAGS.dnn_hidden_units.split(",")]
    model = MLP(32 * 32 * 3, n_hidden, 10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
    epochs, train_acc, train_loss, test_acc, test_loss = [], [], [], [], []
    for epoch in tqdm(range(FLAGS.max_steps)):
        model.train()
        total_acc, total_loss = 0, 0
        for it, (input, target) in enumerate(train_loader):
            optimizer.zero_grad()
            input = input.reshape(input.shape[0], -1).to(device)
            target = target.to(device)
            pred = model(input)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            total_acc += (torch.argmax(pred, dim=1) == target).sum().item()
            total_loss += loss.item()
        if epoch % FLAGS.eval_freq == 0 or epoch == FLAGS.max_steps - 1:
            epochs.append(epoch)
            model.eval()
            train_acc.append(total_acc / len(train_loader.dataset))
            train_loss.append(total_loss / len(train_loader))

            total_acc, total_loss = 0, 0
            for it, (input, target) in enumerate(test_loader):
                input = input.reshape(input.shape[0], -1).to(device)
                target = target.to(device)
                pred = model(input)
                loss = loss_fn(pred, target)

                total_acc += (torch.argmax(pred, dim=1) == target).sum().item()
                total_loss += loss.item()
            test_acc.append(total_acc / len(test_loader.dataset))
            test_loss.append(total_loss / len(test_loader))

            print(
                f"Epoch {epoch} | Train Acc {train_acc[-1]:.4f} | Train Loss {train_loss[-1]:.4f} | Test Acc {test_acc[-1]:.4f} | Test Loss {test_loss[-1]:.4f}"
            )
    return epochs, train_acc, train_loss, test_acc, test_loss


def main(FLAGS, device):
    """
    Main function
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = torchvision.datasets.CIFAR10(
        root="../data/", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=0
    )
    dataset = torchvision.datasets.CIFAR10(
        root="../data/", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=0
    )
    return train(FLAGS, train_loader, test_loader, device)


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dnn_hidden_units",
        type=str,
        default=DNN_HIDDEN_UNITS_DEFAULT,
        help="Comma separated list of number of units in each hidden layer",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE_DEFAULT,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=MAX_EPOCHS_DEFAULT,
        help="Number of epochs to run trainer.",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=EVAL_FREQ_DEFAULT,
        help="Frequency of evaluation on the test set",
    )
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS, "cpu")
