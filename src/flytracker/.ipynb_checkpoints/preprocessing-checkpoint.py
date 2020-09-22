import cv2 as cv
import numpy as np

class OriginalPreprocessing:
    def __init__(self, min_threshold=None, dilation_kernel=None):
        self.min_threshold = min_threshold
        if dilation_kernel is None:
            self.dilation_kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

    def __call__(self, frame):
        # Step 1 - Preprocssing: turning into grayscale
        # TO DO: might be faster to do this with ffmpeg?
        frame_grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Step 2 - thresholding, 
        # TO DO: include automatic minimum threshold
        if self.min_threshold is None:
            self.min_threshold = self.find_minimum_threshold(frame_grayscale)
        _, thresholded_frame = cv.threshold(frame_grayscale, self.min_threshold, 255, cv.THRESH_BINARY_INV)
    
        #Step 3 - Post Processing: Dilating and blurring
        
        #processed_frame = cv.dilate(thresholded_frame, self.dilation_kernel)
        #processed_frame = cv.medianBlur(processed_frame, 3)

        # Step 4 - removing QR codes,
        # TO DO: make automatic, right now its hardcoded
        processed_frame = thresholded_frame.copy()

        processed_frame[:200, :300] = 0 # upper left corner
        processed_frame[-250:, :300] = 0 # lower left corner
        processed_frame[-250:, -300:] = 0 # lower right corner
        processed_frame[:200, -300:] = 0 # lower right corner

        #any weird shit outside of the roi we remove as well
        processed_frame[:70, :] = 0
        processed_frame[-120:, :] = 0
        processed_frame[:, :180] = 0
        processed_frame[:, -200:] = 0

        return processed_frame

    def find_minimum_threshold(self, frame, margin=5):
        detector = cv.SimpleBlobDetector_create(self.blob_detector_params) # which params...
        ex_mat = []
        keypoints = detector.detect(frame)

        height, width = frame.shape
        if len(keypoints) != 0:
            for kpoint in keypoints:
                x = int(kpoint.pt[0])
                y = int(kpoint.pt[1])
                if x > 2*margin and y > 2*margin and x < width - 2*margin and y < height - 2*margin:
                    ex_mat.append(np.reshape(frame[y-margin:y+margin,x-margin:x+margin],(1,100)))
            minimum_threshold = int(np.median(ex_mat)-0.8*np.std(ex_mat))

        else:
            minimum_threshold = 40
        #print(f'Minimum threshold: {minimum_threshold}')
        return minimum_threshold

    @property
    def blob_detector_params(self):
        # Setup SimpleBlobDetector parameters.
        params = cv.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 15;
        params.maxThreshold = 110;

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 15
        params.maxArea = 80
        params.minDistBetweenBlobs = 5

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
        #self.params.maxCircularity = 0.49

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.25

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        params.maxInertiaRatio = 0.2

        return params