import generators as gn
import numpy as np
from sklearn.cluster import KMeans as KMeansSK


def distance_matrix(matrix_1: np.array, matrix_2: np.array) -> np.ndarray:
    """Calculates euclidean distance elementwise between two matrix
    Returns:
        np.ndarray
    """
    dm = np.zeros((len(matrix_1), len(matrix_2)))
    for i, el1 in enumerate(matrix_1):
        for j, el2 in enumerate(matrix_2):
            dm[i, j] = (((el1 - el2) ** 2).sum()) ** 0.5
    return dm


class KMeans:
    def __init__(self, K, n_iter):
        self.K = K
        self.n_iter = n_iter
        self.means: np.ndarray
        self.history = []

    def fit(self, points, distance="euclidean"):
        self.means = points[np.random.choice(points.shape[0], self.K, replace=False)]
        for _ in range(self.n_iter):
            self.history.append(np.copy(self.means))
            assigned_clusters = self.predict(points)
            for j in range(self.K):
                self.means[j, :] = points[assigned_clusters == j].mean(axis=0)

    def predict(self, points):
        dm = distance_matrix(points, self.means)
        return dm.argmin(axis=-1)

    @property
    def cluster_centers_(self):
        return self.means


if __name__ == "__main__":
    data, means = gn.generate_clusters(3, 2, [500, 200, 200])
    kmeans = KMeans(3, 100)
    kmeans.fit(data)
    skmeans = KMeansSK(n_clusters=3, random_state=0).fit(data)

    # plt.scatter(data[:, 0], data[:, 1])
    # for p in kmeans.history[0]:
    #     plt.plot(p[0], p[1], 'kx')
    # # for m in kmeans.history:
    # #     for p in m:
    # #         plt.plot(p[0], p[1], 'ro')
    # for p in kmeans.means:
    #     plt.plot(p[0], p[1], 'rx')

    # for p in means:
    #     plt.plot(p[0], p[1], 'bx')
    # for p in skmeans.cluster_centers_:
    #     plt.plot(p[0], p[1], 'gx')

    # plt.show()
    # test_data = np.array([[15, 13]])
    # resp = kmeans.predict(test_data)[0]
    # respskmeans = skmeans.predict(test_data)[0]
    # print(kmeans.cluster_centers_[resp],
    #       skmeans.cluster_centers_[respskmeans])
