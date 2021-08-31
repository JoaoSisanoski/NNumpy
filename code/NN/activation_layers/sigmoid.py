import numpy as np
from activation_layers.activation import ActivationLayer


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)


class Sigmoid(ActivationLayer):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)
