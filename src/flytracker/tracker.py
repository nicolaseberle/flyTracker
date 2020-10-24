import numpy as np
import cv2 as cv
import pandas as pd
from sklearn.cluster import KMeans
from os import path


class Tracker:
    def __init__(self, movie_path, mask, output_path):
        self.movie_path = movie_path
        self.output_path = output_path

        self.mask = mask
        if self.mask is not None:
            self.mask = self.mask.astype('uint8')

        self.n_flies = None
        self.initial_frame = None

    def run(self, n_frames, n_initialize):
        capture = cv.VideoCapture(self.movie_path)
        if self.mask is None:
            self.mask = np.ones((int(capture.get(4)), int(capture.get(3))), dtype='uint8')

        self.n_flies, initial_positions, self.initial_frame = self.initialize(capture, n_initialize, self.mask)     

        locations = self.localize(capture, self.mask, self.n_flies, initial_positions, n_frames)  # Localizing flies
        dataset = self.post_process(locations, self.initial_frame)  # Postprocessing

        output_path = path.join(self.output_path, 'df.hdf')
        dataset.to_hdf(output_path, 'df')
        return dataset

    def initialize(self, capture, n_frames, mask):
        """
        Runs a blob detector on the first n_frames and determines the number of flies and builds the estimator.
        """
        # Run blob detector at the first n_frames to get number of flies and initial position
        blob_detector = cv.SimpleBlobDetector_create(self.default_blob_detector_params)

        n_blobs = []
        for frame_idx in np.arange(2 * n_frames):
            image = cv.cvtColor(capture.read()[1], cv.COLOR_BGR2GRAY)
            keypoints = blob_detector.detect(image * mask)  # get keypoints
            n_blobs.append(len(keypoints))

            if len(n_blobs) == n_frames:
                n_flies = int(np.median(n_blobs)) # we define number of flies as median number found over first n frames
            if (len(n_blobs) >= n_frames) and (n_blobs[-1] == n_flies):
                locations = np.array([keypoint.pt for keypoint in keypoints])
                initial_frame = frame_idx
                break

        return n_flies, locations, initial_frame

    def localize(self, capture, mask, n_flies, initial_position, n_frames):
        locations = [initial_position]
        estimator = KMeans(n_clusters=n_flies, n_init=1) # we do a lot the first time to make sure we get it right
        
        for _ in np.arange(n_frames):
            # Load as grayscale, apply mask, find nonzero, all inplace for speed,
            ret, image = capture.read()
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            fly_pixels = cv.findNonZero(cv.bitwise_and((image < 120).astype('uint8'), mask)).squeeze() 

            # Set initial centroids as previous frame' locations and do kmeans
            estimator.init = locations[-1]
            locations.append(estimator.fit(fly_pixels).cluster_centers_)
        return locations

    def post_process(self, locations, initial_frame):
        n_frames = len(locations)
        n_flies = len(locations[0])
        identities = (np.arange(n_flies)[None, :] * np.ones((n_frames, n_flies))).reshape(-1, 1) # we get free tracking from the kmeans
        frames = (np.arange(initial_frame, n_frames + initial_frame)[:, None] * np.ones((n_frames, n_flies))).reshape(-1, 1)
        df = pd.DataFrame(np.concatenate([frames, identities, np.concatenate(locations, axis=0)], axis=1), columns=['frame', 'ID', 'x', 'y'])
        return df

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