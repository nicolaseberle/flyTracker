import numpy as np
import cv2 as cv


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


def localize_blob(image: np.ndarray, blob_detector_params, preprocessor) -> np.ndarray:
    """Find flies using blob detector"""
    blob_detector = cv.SimpleBlobDetector_create(blob_detector_params)
    keypoints = blob_detector.detect(
        preprocessor(image).numpy().squeeze()
    )  # get keypoints
    return np.array([keypoint.pt for keypoint in keypoints])

