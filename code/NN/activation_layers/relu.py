import numpy as np
from activation_layers.activation import ActivationLayer


def relu(x):
    return np.maximum(x, 0)


def relu_prime(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


class ReLU(ActivationLayer):
    def __init__(self):
        super().__init__(relu, relu_prime)
