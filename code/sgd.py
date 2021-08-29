import numpy as np


class GradientDescent:
    def __init__(self, max_iter=100, tol=1e-3, lr=0.01):
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr

    def dw_dl(self, error, X):
        """DW_DL is the derivative of the MSE in regards to the
        weights.
        """
        # dot = X*error
        dot = X.T * error
        gradient = -2 * (dot).mean(axis=1)
        return gradient

    def db_dl(self, error):
        """db_dl is the derivative of the MSE in regards to the
        bias.
        """
        return -2 * (error.mean())

    def predict(self, X):
        return X.dot(self.weights) + self.bias

    def residuals(self, X, y):
        prediction = self.predict(X)
        return y - prediction

    def fit(self, X, y):
        self.weights = rng.randn(len(X[0]))
        self.bias = rng.randn(1)
        for i in range(self.max_iter):
            error = self.residuals(X, y)
            db_dl = self.db_dl(error)
            dw_dl = self.dw_dl(error, X)
            self.weights = self.weights - self.lr * dw_dl
            self.bias = self.bias - self.lr * db_dl
            (error ** 2).mean()


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n_samples, n_features, noise_level = 300, 30, 3
    X = rng.rand(n_samples, 1) * 5
    bias = rng.rand(1) * 10
    y = 30 * X + bias
    y = y + rng.randn(n_samples, 1) * noise_level
    y = y[:, 0]
    sgd = GradientDescent()
    sgd.fit(X, y)
