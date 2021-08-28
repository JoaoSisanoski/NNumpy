import random
from typing import List, Optional, Tuple

import numpy as np


def generate_clusters(
    K: int,
    n_features: int,
    n_samples: List[int],
    cluster_std: Optional[int] = 3,
    means_scale: Optional[int] = 10,
    shuffle: Optional[bool] = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns K random generated clusters with D dimensions.
    n_samples is the amount of points per clusters: e.g [100, 200, 300]
    """
    assert K == len(n_samples)
    means = np.random.randn(K, n_features) * means_scale
    samples = []
    # TODO change later to use np.array instead of list
    for mean, n_sample in zip(means, n_samples):
        new_points = (
            np.random.randn(n_sample, n_features) * cluster_std + mean
        ).tolist()
        samples += new_points
    if shuffle:
        random.shuffle(samples)
    return np.array(samples), means
