import cv2 as cv 
import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans


class Frame:
    """Class for each frame of the movie containing active area mask
    and can return grayscale, apply mask etc. A lot easier to have a single
    object instead of frame_grayscale etc."""

    def __init__(self, image, mask):
        self.mask = mask
        self.image = image

    def __call__(self, gray=True, masked=True):
        """ Returns grayscale masked image."""
        image = self.image
        if gray is True:
            image = self.grayscale(image) 
        if (masked is True) and (gray is True):
            image = self.apply_mask(image, self.mask) 
        return image

    def grayscale(self, image):
        """Returns grayscale of image."""
        grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return grayscale_image

    def apply_mask(self, image, mask):
        """Returns masked image, only works on grayscale image"""
        masked_image = image
        np.putmask(masked_image, ~mask, 255)
        return masked_image


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


class HungarianTracker:
    def __init__(self, threshold=50):
        self.threshold = threshold

    def __call__(self, coordinates_i, coordinates_j):
        dist_matrix = distance_matrix(coordinates_i, coordinates_j)
        dist_matrix[dist_matrix >= self.threshold] = 1e4 # distances longer than threshold can't happen

        identities_j = linear_sum_assignment(dist_matrix)[1].squeeze()
        return identities_j


class KMeansCorrect:
    def __init__(self, n_flies):
        self.n_flies = n_flies
        self.estimator = KMeans(n_clusters=self.n_flies, init=np.zeros((n_flies, 2)), n_init=1)
        
    def __call__(self, image, previous_frame_locations):
        # We first threshold
        thresholded_frame = cv.threshold(image(), 120, 255, cv.THRESH_BINARY_INV)[1]
        
        # Get the location of the non-zero pixels
        fly_pixels = np.stack(np.where(thresholded_frame != 0)).T[:, ::-1] # to get y and x good
        
        # Fit and get cluster centres
        self.estimator.init = previous_frame_locations  # initialize with location from previous frame
        self.estimator.fit(fly_pixels)
        locations = self.estimator.cluster_centers_
        
        return locations
