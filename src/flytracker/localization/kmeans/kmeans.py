import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from .kmeans_torch import kmeans_torch
import torch


def localize_kmeans(
    image: np.ndarray, prev_locations: np.ndarray, threshold: int = 120, tol=1e-4
) -> np.ndarray:
    """Find flies using kmeans."""
    init = prev_locations[-1]
    n_flies = init.shape[0]
    fly_pixels = cv.findNonZero(
        (image.numpy().squeeze() < threshold).astype("uint8")
    ).squeeze()
    locations = (
        KMeans(n_clusters=n_flies, n_init=1, init=init, tol=tol)
        .fit(fly_pixels)
        .cluster_centers_
    )
    return prev_locations + [locations]


def localize_kmeans_torch(threshold=120, tol=1e-4, device="cuda"):
    def localize(image, prev_location):
        fly_pixels = torch.fliplr(torch.nonzero(image < threshold).type(torch.float32))
        locations, _ = kmeans(fly_pixels, prev_location)
        return locations

    kmeans = kmeans_torch(tol=tol, device=device)
    return localize

