import cv2 as cv
import numpy as np
from itertools import chain


class OriginalLocalising:
    def __init__(self, n_flies=40):
        self.n_flies = n_flies
        self.margin = 5

    def __call__(self, thresholded_frame, frame):
        # Finding contours
        contours = cv.findContours(thresholded_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
        # Step 6 - iterating over flies
        fly_features = []
        arc_lengths = []
        for contour in contours:
            ellipse = cv.fitEllipse(contour)
            arc_lengths.append(cv.arcLength(contour, closed=True))
            ellipse = list(chain(*(feature if isinstance(feature, tuple) else (feature,) for feature in ellipse))) # flattening data
            fly_features.append(ellipse)
        
        if len(fly_features) != self.n_flies:
            # Finding bbox of maximum arc length
            max_arc_idx = np.argmax(arc_lengths)
            bbox = cv.boundingRect(contours[max_arc_idx])
            
            # Doing watershed to separate
            markers = self.watershed(thresholded_frame[bbox[1] - self.margin:bbox[1] + bbox[3] + self.margin, bbox[0] - self.margin:bbox[0] + bbox[2] + self.margin], frame[bbox[1] - self.margin:bbox[1] + bbox[3] + self.margin, bbox[0] - self.margin:bbox[0] + bbox[2] + self.margin])
            
            # Updating contours
            local_contours = cv.findContours(markers, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
            
            # deleting the old marker
            contours.pop(max_arc_idx)
            fly_features.pop(max_arc_idx)
            # Adding new contours and ellipses
            contours.extend(local_contours)
            for contour in local_contours:
                ellipse = cv.fitEllipse(contour.astype(np.float32))
                ellipse_flattened = list(chain(*(feature if isinstance(feature, tuple) else (feature,) for feature in ellipse))) # flattening data
                fly_features.append(ellipse_flattened)

        return np.array(fly_features)

    def watershed(self, thresholded_frame, frame):
        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(thresholded_frame, cv.MORPH_OPEN, kernel, iterations=2)
        # sure background area
        sure_bg = cv.dilate(opening, kernel, iterations=3)
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        markers[markers > 1] = 2

        markers = cv.watershed(frame, markers)
        markers[markers == -1] = 1
        markers -= 1
        
        return markers.astype(np.uint8)

