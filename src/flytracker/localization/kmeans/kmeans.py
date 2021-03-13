import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from .kmeans_jax import kmeans_jax
from .kmeans_torch import kmeans_torch
import torch


# Blob detector
def localize_kmeans(
    image: np.ndarray, init: np.ndarray, threshold: int = 120
) -> np.ndarray:
    """Find flies using kmeans."""
    n_flies = init.shape[0]
    fly_pixels = cv.findNonZero((image < threshold).astype("uint8")).squeeze()
    locations = (
        KMeans(n_clusters=n_flies, n_init=1, init=init).fit(fly_pixels).cluster_centers_
    )
    return locations


def localize_kmeans_jax(
    image: np.ndarray, init: np.ndarray, threshold: int = 120
) -> np.ndarray:
    """Find flies using kmeans."""
    fly_pixels = cv.findNonZero((image < threshold).astype("uint8")).squeeze()
    locations = kmeans_jax(fly_pixels, init)[0]
    return locations


def localize_kmeans_torch(loader, init, n_frames=900, threshold=120):
    """Find flies using blob detector and
    calulate number of flies."""
    data = [init]
    for frame_idx, frame in enumerate(loader):
        frame = frame.cuda(non_blocking=True)
        fly_pixels = torch.nonzero(frame.squeeze() < threshold).type(torch.float32)
        data.append(kmeans_torch(fly_pixels, data[-1]))

        if frame_idx == n_frames:
            break
        if frame_idx % 100 == 0:
            print(f"Done with {frame_idx}")
    return data

