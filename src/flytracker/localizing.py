import cv2 as cv
import numpy as np


class OriginalLocalising:
    def __init__(self, max_perimeter=200, max_flies_per_cluster=5):
        self.max_perimeter = max_perimeter
        self.max_flies_per_cluster = max_flies_per_cluster

    def __call__(self, frame):
        # Finding contours
        contours = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    
        # Step 6 - iterating over flies
        for contour in contours:
            # Get rectangular bounding boxes (why two?)
            bbox = cv.minAreaRect(contour) 
            x, y, w, h = cv.boundingRect(contour)

            # Check if perimeter isnt too long
            perimeter = cv.arcLength(contour, True)
            if perimeter > self.max_perimeter:
                print('Perimeter too long. Skipping.')
                continue
    
            # Get area and coordinates of bounding box
            area = cv.contourArea(contour)
            bbox_coors = cv.boxPoints(bbox).astype(np.int)

            # Try and fit an ellipse
            if contour.shape[0] >= 5:
                ellipse = cv.fitEllipse(contour)
            else:
                ellipse = [(-1, -1), (-1, -1), -1]

            # Finding pixels of fly
            mask = cv.fillPoly(np.zeros_like(frame), [contour], 1) # oof this is a massive array for a single fly...
            pixels = cv.findNonZero(mask[y:(y + h), x:(x + w)]).astype(np.float32)
            n_pixels = pixels.shape[0]
        
            # WTF is happening below? No idea....
            centroids = []
            kmeans_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            flags = cv.KMEANS_RANDOM_CENTERS
            for n in range(2, self.max_flies_per_cluster + 1): #iterating over what exactly?
                if n_pixels <= n: # what does the number of pixels have to do with number of flies?
                    cx = ((bbox_coors[0][0] + bbox_coors[1][0] + bbox_coors[2][0] + bbox_coors[3][0]) / 4) # why / 4 ?
                    cy = ((bbox_coors[0][1] + bbox_coors[1][1] + bbox_coors[2][1] + bbox_coors[3][1]) / 4)
                    for num in range(n):
                        centroids.append(int(cx))
                        centroids.append(int(cy))
                        centroids.append(n)
            else:
                ret, label, center = cv.kmeans(pixels, n, None, kmeans_criteria, 10, flags)
                for num in range(n):
                    centroids.append(int(float(center[num][0]) + x))
                    centroids.append(int(float(center[num][1]) + y))
                    centroids.append(np.sum(label == num))

        # centres of the contour and angle of bbox
        # in original code there's a bug in the angle, its converted to degrees twice (once here, once in current_plot).
        # We only do it here.
        cx = np.mean(pixels, 0)[0, 0] + x
        cy = np.mean(pixels, 0)[0, 1] + y
        angle = np.arctan((bbox_coors[0, 1]-bbox_coors[1, 1]) / (bbox_coors[0, 0]-bbox_coors[1, 0]))*180/np.pi 

        # Collecting output
        centroids = np.array(centroids, dtype=np.float32).reshape(1, 1, -1)
        locations = np.array([[[cx, cy, angle, area, ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2]]]], dtype=np.float32)
        locations = np.concatenate((locations, centroids), axis=2)
    
        return locations

