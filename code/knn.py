import numpy as np


class KNearestNeighbours:
    def __init__(self, n_neighbours):
        self.n_neighbours = n_neighbours
        self.X: np.ndarray
        self.y: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
