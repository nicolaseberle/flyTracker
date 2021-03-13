from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
import numpy as np


def hungarian(locs_prev: np.ndarray, locs_new: np.ndarray) -> np.ndarray:
    """Returns ordered fly location (i.e. tracks)"""
    new_ordering = linear_sum_assignment(
        distance_matrix(locs_new[:, :2], locs_prev[:, :2])
    )[1]
    return locs_new[new_ordering]
