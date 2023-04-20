import argparse
import numpy as np
from pytorch_mlp import MLP
from sklearn.datasets import make_moons, make_circles, make_blobs
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = "20"
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 100
EVAL_FREQ_DEFAULT = 10

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
    return (
        torch.argmax(predictions, dim=1) == torch.argmax(targets, dim=1)
    ).sum() / predictions.shape[0]


def train(FLAGS, dataset, device):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    data, label = dataset
    label = F.one_hot(torch.from_numpy(label)).float().to(device)
    data = torch.from_numpy(data).float().to(device)
    train_size, test_size = 800, data.shape[0] - 800
    train_data, test_data = data[:train_size], data[train_size:]
    train_label, test_label = label[0:train_size], label[train_size:]

    plt.figure(1, figsize=(20, 5))
    plt.subplot(121)
    plt.scatter(
        train_data[:, 0].cpu(),
        train_data[:, 1].cpu(),
        c=dataset[1][:train_size],
        label="Train Samples",
    )
    plt.title(FLAGS.dataset + " dataset training samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.subplot(122)
    plt.scatter(
        test_data[:, 0].cpu(),
        test_data[:, 1].cpu(),
        c=dataset[1][train_size:],
        label="Test Samples",
    )
    plt.title(FLAGS.dataset + " dataset testing samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    n_hidden = [int(num) for num in FLAGS.dnn_hidden_units.split(",")]
    model = MLP(2, n_hidden, 2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
    epochs, train_acc, train_loss, test_acc, test_loss = [], [], [], [], []
    for epoch in tqdm(range(FLAGS.max_steps)):
        model.train()
        for i in range(len(train_data)):
            optimizer.zero_grad()
            input = train_data[i]
            label = train_label[i]
            pred = model(input)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
        if epoch % FLAGS.eval_freq == 0 or epoch == FLAGS.max_steps - 1:
            epochs.append(epoch)
            model.eval()
            train_pred = model(train_data)
            loss = loss_fn(train_pred, train_label)
            train_acc.append(accuracy(train_pred, train_label))
            train_loss.append(loss.item())

            test_pred = model(test_data)
            loss = loss_fn(test_pred, test_label)
            test_acc.append(accuracy(test_pred, test_label))
            test_loss.append(loss.item())

            print(
                f"Epoch {epoch} | Train Acc {train_acc[-1]:.4f} | Train Loss {train_loss[-1]:.4f} | Test Acc {test_acc[-1]:.4f} | Test Loss {test_loss[-1]:.4f}"
            )
    return epochs, train_acc, train_loss, test_acc, test_loss


def main(FLAGS, device):
    """
    Main function
    """
    if FLAGS.dataset == "make_moon":
        dataset = make_moons(n_samples=1000, shuffle=True)
    elif FLAGS.dataset == "make_circles":
        dataset = make_circles(n_samples=1000, shuffle=True)
    elif FLAGS.dataset == "make_classification":
        dataset = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=0)
    return train(FLAGS, dataset, device)


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
    parser.add_argument(
        "--dataset", type=str, default="make_moon", help="Dataset type."
    )
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS, 'mps')
