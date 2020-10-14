import numpy as np
import cv2 as cv
from sklearn import cluster


class KMeans:
    def __init__(self, n_flies, lower_threshold=120, **kmeans_kwargs):
        self.n_flies = n_flies
        self.lower_threshold = lower_threshold
        self.estimator = cluster.KMeans(n_clusters=self.n_flies, **kmeans_kwargs)

    def __call__(self, image, previous_frame_locations):
        # We first threshold
        thresholded_frame = cv.threshold(image, self.lower_threshold, 255, cv.THRESH_BINARY_INV)[1]

        # Get the location of the non-zero pixels
        fly_pixels = np.stack(np.where(thresholded_frame != 0)).T[:, ::-1] # to get y and x good

        # Fit and get cluster centres
        if previous_frame_locations is None:
            self.estimator.n_init = 10
            self.estimator.fit(fly_pixels)
        else:
            self.estimator.n_init = 1
            self.estimator.init = previous_frame_locations
            self.estimator.fit(fly_pixels)

        return self.estimator.cluster_centers_


class BlobDetector:
    def __init__(self, params=None):
        if params is None:
            self.blob_detector = cv.SimpleBlobDetector_create(self.default_blob_detector_params)
        else:
            self.blob_detector = cv.SimpleBlobDetector_create(params)

    def __call__(self, image):
        keypoints = self.blob_detector.detect(image(gray=True, masked=True))  # get keypoints
        coordinates = np.array([keypoint.pt for keypoint in keypoints])  # extract coordinates from keypoints
        return coordinates

    @property
    def default_blob_detector_params(self):
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
