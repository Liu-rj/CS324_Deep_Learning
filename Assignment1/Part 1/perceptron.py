import numpy as np


class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=100, learning_rate=1e-2):
        """
        Initializes perceptron object.
        Args:
            n_inputs: number of inputs.
            max_epochs: maximum number of training cycles.
            learning_rate: magnitude of weight changes at each training cycle
        """
        self.n_inputs = n_inputs
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.w = np.zeros((n_inputs + 1, 1), dtype=np.float32)

    def forward(self, input):
        """
        Predict label from input 
        Args:
            input: array of dimension equal to n_inputs.
        """
        expanded_input = np.concatenate(
            [input, np.ones((input.shape[0], 1))], axis=1)
        label = np.sign((expanded_input @ self.w).reshape(-1))
        return label

    def train(self, training_inputs, labels):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            labels: arrays of expected output value for the corresponding point in training_inputs.
        """
        expanded_input = np.concatenate(
            [training_inputs, np.ones((training_inputs.shape[0], 1))], axis=1)
        idx = np.arange(expanded_input.shape[0])
        for epoch in range(self.max_epochs):
            permuted_idx = np.random.permutation(idx)
            train_set = expanded_input[permuted_idx]
            train_labels = labels[permuted_idx]
            for sample, label in zip(train_set, train_labels):
                sample = np.expand_dims(sample, 0)
                if label * (sample @ self.w) <= 0:
                    self.w += self.learning_rate * \
                        label * sample.T
