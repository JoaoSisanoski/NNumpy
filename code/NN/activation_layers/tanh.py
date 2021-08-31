import numpy as np
from activation_layers.activation import ActivationLayer


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - tanh(x) ** 2


class Tanh(ActivationLayer):
    def __init__(self):
        super().__init__(tanh, tanh_prime)
