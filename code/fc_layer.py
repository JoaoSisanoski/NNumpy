import numpy as np
from layer import Layer


class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(1, output_size)

    def forward(self, input_data):
        self.input = input_data
        self.output = input_data.dot(self.weights) + self.bias
        return self.output

    def backward(self, error, lr):
        input_error = np.dot(error, self.weights.T).squeeze()
        weights_error = np.dot(self.input.T, error).mean()
        self.weights -= lr * weights_error
        self.bias -= lr * error
        return input_error
