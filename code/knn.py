import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from utils import distance_matrix


class KNearestNeighbours:
    def __init__(self, n_neighbours):
        self.n_neighbours = n_neighbours
        self.X: np.ndarray
        self.y: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def predict(self, points: np.ndarray):
        dm = distance_matrix(self.X, points)
        closest = np.zeros(len(points))
        for i, _ in enumerate(points):

            min_elements = np.argpartition(dm[:, i], self.n_neighbours)[
                : self.n_neighbours
            ]
            closest_ys = self.y[min_elements]
            # cnt = Counter(closest_ys)
            # return cnt.most_common(1)[]
            bins = np.bincount(closest_ys)
            closest[i] = np.argmax(bins)
        return closest


if __name__ == "__main__":
    X, y = make_classification(100, 10, n_classes=3, n_informative=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    knn = KNearestNeighbours(4)
    knn.fit(X_train, y_train)
    cl = knn.predict(X_test)
    knn_sklearn = KNeighborsClassifier(n_neighbors=4)
    knn_sklearn.fit(X_train, y_train)
    cl_sklearn = knn_sklearn.predict(X_test)
    # print(cl)
    print(cl_sklearn, cl)
