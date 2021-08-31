from typing import List

import numpy as np
from activation_layers import ReLU
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
            for x, y in zip(x_train, y_train):
                err = 0
                output = x
                for layer in self.layers:
                    output = layer.forward(output)
                err += self.loss_function(y, output)
                error = self.gradient(y, output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            err /= samples
            print("epoch %d/%d   error=%f" % (i + 1, epochs, err))


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true)


if __name__ == "__main__":
    from dense import Dense

    rng = np.random.RandomState(0)
    n_samples, n_features, noise_level = 500, 30, 3
    X = rng.rand(n_samples, 1) * 5
    bias = rng.rand(1) * 10
    y = 30 * X + bias
    y = y + rng.randn(n_samples, 1) * noise_level

    net = Network()
    net.add(Dense(1, 1))
    net.add(ReLU())
    # net.add(Dense(4, 1))
    # net.add(Tanh())
    # train
    net.compile(mse, mse_prime)
    net.fit(X[:, np.newaxis], y, epochs=1000, learning_rate=0.01)

    # test
    pred = X[:10]
    y = y[:10]
    out = net.predict(pred)
    for y_pred, y_true in zip(out, y):
        print(y_pred, y_true)
