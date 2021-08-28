import numpy as np


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
