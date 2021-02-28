import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from .kmeans import kmeans_jax, kmeans_torch
import torch

# Blob detector
def default_blob_detector_params() -> object:
    """Blob detector params used to perform initial localization."""
    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 20
    params.maxThreshold = 150

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 15
    params.maxArea = 60
    params.minDistBetweenBlobs = 1.0

    # Turn off other filters
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    return params


def blob_detector_localization(image: np.ndarray) -> np.ndarray:
    """Find flies using blob detector"""
    blob_detector = cv.SimpleBlobDetector_create(default_blob_detector_params())
    keypoints = blob_detector.detect(image)  # get keypoints
    return np.array([keypoint.pt for keypoint in keypoints])


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
    return data


def hungarian(locs_new: np.ndarray, locs_prev: np.ndarray) -> np.ndarray:
    """Returns ordered fly location (i.e. tracks)"""
    new_ordering = linear_sum_assignment(
        distance_matrix(locs_new[:, :2], locs_prev[:, :2])
    )[
        1
    ]  # Distance matrix only over position
    return locs_new[new_ordering]


def initialize(loader, n_frames=100):
    """Find flies using blob detector and
    calulate number of flies."""
    n_blobs = []

    for frame_idx, frame in enumerate(loader):
        locations = blob_detector_localization(frame.numpy().squeeze())
        n_blobs.append(locations.shape[0])
        if len(n_blobs) >= n_frames:
            n_flies = int(np.median(n_blobs))
            if n_blobs[-1] == n_flies:
                break
    # pluse on cause the next one is the first
    initial_frame = frame_idx + 1
    return n_flies, locations, initial_frame
