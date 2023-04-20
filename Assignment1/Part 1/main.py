import numpy as np
from perceptron import *

np.random.seed(321)

# Task 1: generate datasets
data1 = np.random.multivariate_normal([0, 0], [[2, 0], [0, 2]], 100)
data2 = np.random.multivariate_normal([10, 10], [[3, 0], [0, 5]], 100)
train_set = np.concatenate([data1[:80], data2[:80]])
test_set = np.concatenate([data1[80:], data2[80:]])
train_label = np.array([-1] * 80 + [1] * 80)
test_label = np.array([-1] * 20 + [1] * 20)

# Task 2: implement perceptron, see perceptron.py

# Task 3: train the perceptron and compute test accuracy
model = Perceptron(train_set.shape[1])
model.train(train_set, train_label)
pred = model.forward(test_set)
print('test acc:', (pred == test_label).sum() / pred.shape[0])

# Task 4: see task4.ipynb
