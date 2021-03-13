from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
import numpy as np


def hungarian(locs_new: np.ndarray, locs_prev: np.ndarray) -> np.ndarray:
    """Returns ordered fly location (i.e. tracks)"""
    new_ordering = linear_sum_assignment(
        distance_matrix(locs_new[:, :2], locs_prev[:, :2])
    )[
        1
    ]  # Distance matrix only over position
    return locs_new[new_ordering]
