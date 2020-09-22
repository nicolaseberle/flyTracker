import cv2 as cv 
import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix


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


class Tracker:
    def __call__(self, coordinates_i, coordinates_j, identities_i):
        identities_j = linear_sum_assignment(distance_matrix(coordinates_i, coordinates_j))[1]
        return identities_i[identities_j][:, None]
