import numpy as np
import cv2 as cv
import pandas as pd
from sklearn.cluster import KMeans


class Tracker:
    def __init__(self, n_flies, mask):
        self.n_flies = n_flies
        self.mask = mask.astype('uint8')
        self._estimator = KMeans(n_clusters=self.n_flies, n_init=20)

    def run(self, path, n_frames):
        locations = self._localize(path, n_frames)  # Localizing flies
        dataset = self._post_process(locations)  # Postprocessing

        return dataset

    def _localize(self, path, n_frames):
        locations = []

        capture = cv.VideoCapture(path)
        for frame_idx in np.arange(n_frames):
            # Load as grayscale, apply mask, find nonzero, all inplace for speed
            image = cv.cvtColor(capture.read()[1], cv.COLOR_BGR2GRAY)
            fly_pixels = cv.findNonZero(cv.bitwise_and((image < 120).astype('uint8'), self.mask)).squeeze() 

            # Set initial centroids as previous frame' locations and do kmeans
            if frame_idx >= 1:
                self._estimator.n_init = 1
                self._estimator.init = locations[-1]
            locations.append(self._estimator.fit(fly_pixels).cluster_centers_)
        return locations

    def _post_process(self, locations):
        n_frames = len(locations)
        n_flies = len(locations[0])
        identities = (np.arange(n_flies)[None, :] * np.ones((n_frames, n_flies))).reshape(-1, 1)
        frames = (np.arange(n_frames)[:, None] * np.ones((n_frames, n_flies))).reshape(-1, 1)
        df = pd.DataFrame(np.concatenate([frames, identities, np.concatenate(locations, axis=0)], axis=1), columns=['frame', 'ID', 'x', 'y'])
        return df
