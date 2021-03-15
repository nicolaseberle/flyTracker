import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from .kmeans_torch import kmeans_torch
import torch


# Blob detector
def localize_kmeans(
    image: np.ndarray, init: np.ndarray, threshold: int = 120
) -> np.ndarray:
    """Find flies using kmeans."""
    n_flies = init.shape[0]
    fly_pixels = cv.findNonZero(
        (image.numpy().squeeze() < threshold).astype("uint8")
    ).squeeze()
    locations = (
        KMeans(n_clusters=n_flies, n_init=1, init=init).fit(fly_pixels).cluster_centers_
    )
    return locations


def localize_kmeans_torch(image, prev_locations, threshold=120, device="cuda"):
    initializing = len(prev_locations) == 1
    if initializing:
        prev_locations = [
            torch.tensor(prev_locations[0], dtype=torch.float32).to(device)
        ]

    image, init = image.squeeze().to(device, non_blocking=True), prev_locations[-1]
    fly_pixels = torch.fliplr(torch.nonzero(image < threshold).type(torch.float32))
    locations = kmeans_torch(fly_pixels, init)

    return prev_locations + [locations]

