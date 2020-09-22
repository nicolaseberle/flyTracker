import cv2 as cv
import numpy as np


class OriginalLocalising:
    def __init__(self, n_flies=40, blob_params=None):
        self.n_flies = n_flies
        
        if blob_params is not None:
            self.detector = cv.SimpleBlobDetector_create(blob_params)
        else:
            self.detector = cv.SimpleBlobDetector_create(self.blob_detector_params)
         
    def __call__(self, processed_frame, raw_frame):
        keypoints = self.detector.detect(processed_frame)

        return keypoints

    @property
    def blob_detector_params(self):
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
