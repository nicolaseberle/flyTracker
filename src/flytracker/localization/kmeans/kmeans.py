from sklearn.cluster import KMeans
from .kmeans_torch import kmeans_torch
import torch
import numpy as np


def localize_kmeans_sklearn(threshold=120, tol=1e-4):
    def localize(image, prev_location):
        n_flies = prev_location.shape[0]
        fly_pixels = torch.nonzero(image < threshold).type(torch.float32)
        fit = KMeans(n_clusters=n_flies, n_init=1, init=prev_location, tol=tol).fit(
            fly_pixels
        )
        locations = torch.tensor(fit.cluster_centers_)
        # we do tracking at the end, so we approximate the distance by summing over all the flies.
        fly_space_distance = np.abs(
            np.linalg.norm(locations.mean(axis=0))
            - np.linalg.norm(prev_location.mean(axis=0))
        )
        return locations, fly_space_distance

    return localize


def localize_kmeans_torch(threshold=120, tol=1e-4, device="cuda"):
    def localize(image, prev_location):
        fly_pixels = torch.nonzero(image < threshold).type(torch.float32)
        locations, _ = kmeans(fly_pixels, prev_location)
        # we do tracking at the end, so we approximate the distance by summing over all the flies.

        fly_space_distance = torch.abs(
            torch.linalg.norm(locations.mean(axis=0))
            - torch.linalg.norm(prev_location.mean(axis=0))
        )
        return locations, fly_space_distance

    kmeans = kmeans_torch(tol=tol, device=device)
    return localize
