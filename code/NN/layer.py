class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, _):
        raise NotImplementedError

    def backward(self, _, __):
        raise NotImplementedError
