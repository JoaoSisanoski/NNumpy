import numpy as np
from layer import Layer

# Y = W.X + B

"""
A layer receives an input X and returns an output Y
in the forward function


In the backward function it receives an error in regards to Y
and expects to return an error and regards to X
            _______________________
            |                      |
---> X ---> |                      | ---> Y
            |         layer        |
            |          W           |
<---  de/dx |______________________| <--- de/dy


de/dy = [de/dy1, de/dy2, de/dy3,..., de/dyn]

de/dw =  de/dy(.)X.t

de/db = de/dy

de/dx = W.t(.)de/dy

"""


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input_data):
        """
        weights = [w1, w2, w3, ..., wn]

        input_data = [[x1, x2, x3, ..., xn]]

        resp = x1*w1+x2*w2+x3*w3+...+xn+wn+bias
        """
        self.input = input_data
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, de_dy, lr):
        """


        Args:
            de_dy ([type]): [description]
            lr ([type]): [description]

        Returns:
            [type]: [description]
        """
        weights_gradient = np.dot(de_dy, self.input.T)
        input_gradient = np.dot(self.weights.T, de_dy)
        self.weights -= lr * weights_gradient
        self.bias -= lr * de_dy
        return input_gradient
        # x_t = self.input.T
        # # resp = sum([i*j[0] for i, j in zip(x_t[0], de_dy)])
        # de_dw = np.dot(x_t, de_dy)
        # self.weights -= lr * de_dw
        # nb = (lr * de_dy)
        # self.bias -= nb
        # w_t = self.weights.T
        # de_dx = np.dot(de_dy, self.weights.T)
        # return de_dx
