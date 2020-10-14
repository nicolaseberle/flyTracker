import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans


class Tracker:
    def __init__(self, n_flies, mask):
        self.n_flies = n_flies
        self.mask = mask.astype('uint8')
        self._estimator = KMeans(n_clusters=self.n_flies, n_init=10)
    
    def run(self, path, n_frames):
        locations = self._localize(path, n_frames) # Localizing flies
        #data = self._identify(locations)  # Finding unique identiies
        #dataset = self._post_process(data, n_frames)  # Postprocessing

        return locations

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

    '''
    def _identify(self):

        # Track flies
        if frame_idx > 0:
             identities = [np.arange(self.n_flies)]
            new_frame_identities = tracker(locations[-2], locations[-1])
            locations[-1] = locations[-1][new_frame_identities, :]
            identities.append(np.arange(n_flies))
        
        locations.append(fly_locations)
        time.append(np.ones((n_flies, 1)) * frame_idx)
        dataset = np.concatenate((np.concatenate(time, axis=0), np.concatenate(identities, axis=0)[:, None], np.concatenate(locations, axis=0)), axis=1)
        return dataset

    '''