import numpy as np
import cv2


def default_blob_detector_params() -> object:
    """Blob detector params used to perform initial localization."""
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

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


def localize_blob(blob_detector_params):
    def localize(image):
        # opencv has diffeent axis ordering compared to pytorch nonzero
        # so we flip the column order.
        keypoints = blob_detector.detect(image)  # get keypoints
        return np.fliplr(np.array([keypoint.pt for keypoint in keypoints]))

    blob_detector = cv2.SimpleBlobDetector_create(blob_detector_params)
    return localize

