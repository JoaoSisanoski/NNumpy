from layer import Layer


class ActivationLayer(Layer):
    def __init__(self, activation, gradient):
        self.activation = activation
        self.gradient = gradient

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, _):
        return self.gradient(self.input) * output_error
