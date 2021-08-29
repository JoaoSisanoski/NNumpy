from typing import List

import numpy as np
from layer import Layer


class Network:
    def __init__(self):
        self.layers: List[Layer] = []
        self.loss_function = None
        self.gradient = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, loss_prime):
        self.loss_function = loss
        self.gradient = loss_prime

    def predict(self, input_data):
        samples = len(input_data)
        result = []
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)
        for i in range(epochs):
            err = 0
            output = x_train
            for layer in self.layers:
                output = layer.forward(output)
            err += self.loss_function(y_train, output)
            error = self.gradient(y_train, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, learning_rate)

            err /= samples
            print("epoch %d/%d   error=%f" % (i + 1, epochs, err))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true).mean()


if __name__ == "__main__":
    from fc_layer import FCLayer

    rng = np.random.RandomState(0)
    n_samples, n_features, noise_level = 300, 30, 3
    X = rng.rand(n_samples, 1) * 5
    bias = rng.rand(1) * 10
    y = 30 * X + bias
    y = y + rng.randn(n_samples, 1) * noise_level

    net = Network()
    net.add(FCLayer(1, 3))
    net.add(FCLayer(3, 1))
    # net.add(FCLayer(1, 1))

    net.compile(mse, mse_prime)
    net.fit(X, y, epochs=10, learning_rate=0.1)

    # test
    out = net.predict(X)
