from sklearn.cluster import KMeans
from .kmeans_torch import kmeans_torch
import torch


def localize_kmeans_sklearn(threshold=120, tol=1e-4):
    def localize(image, prev_location):
        n_flies = prev_location.shape[0]
        fly_pixels = torch.nonzero(image < threshold).type(torch.float32)
        fit = KMeans(n_clusters=n_flies, n_init=1, init=prev_location, tol=tol).fit(
            fly_pixels
        )

        return fit.cluster_centers_

    return localize


def localize_kmeans_torch(threshold=120, tol=1e-4, device="cuda"):
    def localize(image, prev_location):
        fly_pixels = torch.nonzero(image < threshold).type(torch.float32)
        locations, _ = kmeans(fly_pixels, prev_location)
        return locations

    kmeans = kmeans_torch(tol=tol, device=device)
    return localize

