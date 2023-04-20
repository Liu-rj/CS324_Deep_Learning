import numpy as np


class Linear(object):
    def __init__(self, in_features, out_features, lr):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
        TODO:
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0. 
        3) Initialize gradients with zeros.
        """
        self.params, self.grads = {}, {}
        self.params['weight'] = np.random.normal(
            loc=0, scale=0.1, size=((in_features, out_features)))
        self.params['bias'] = np.zeros(out_features, dtype=np.float32)
        self.lr = lr

    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside the object
        and use them in the backward pass computation. This is true for *all* forward methods of *all* modules in this class
        """
        out = x @ self.params['weight'] + self.params['bias']
        self.x = x
        return out

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias']. 
        """
        self.grads['weight'] = self.x.T @ dout
        self.grads['bias'] = np.sum(dout, axis=0)
        dx = dout @ self.params['weight'].T
        self.params['weight'] -= self.lr * \
            self.grads['weight'] / self.x.shape[0]
        self.params['bias'] -= self.lr * self.grads['bias'] / self.x.shape[0]
        return dx


class ReLU(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
        """
        out = np.zeros_like(x, dtype=np.float32)
        self.idx = np.where(x > 0)
        out[self.idx] = x[self.idx]
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        dx = np.zeros_like(dout, dtype=np.float32)
        dx[self.idx] = dout[self.idx]
        return dx


class SoftMax(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module

        TODO:
        Implement forward pass of the module. 
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        """
        y = np.exp(x - np.max(x, axis=1, keepdims=True))
        out = y / np.sum(y, axis=1, keepdims=True)
        return out

    def backward(self, dout):
        """
        Backward pass. 
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        # dx = np.zeros_like(dout, dtype=np.float32)
        # for i in range(self.out.shape[0]):
        #     out = self.out[i]
        #     SM = out.reshape((-1, 1))
        #     jac = np.diagflat(out) - np.dot(SM, SM.T)
        #     dx[i] = dout[i] @ jac
        dx = dout
        return dx


class CrossEntropy(object):
    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            out: cross entropy loss
        """
        out = np.mean(np.sum(-y * np.log(x), axis=1))
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            dx: gradient of the loss with respect to the input x.
        """
        dx = x - y
        return dx
