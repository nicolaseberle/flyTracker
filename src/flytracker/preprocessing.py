import cv2 as cv
import numpy as np


class OriginalPreprocessing:
    def __call__(self, frame):
        # Step 1 - Preprocssing: turning into grayscale
        frame_grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Step 2 - Removing QR codes and stuff
        # TO DO: make this automatic
        frame_grayscale[:180, :300] = 255 # upper left corner
        frame_grayscale[-230:, :300] = 255 # lower left corner
        frame_grayscale[-230:, -300:] = 255 # lower right corner
        frame_grayscale[:180, -300:] = 255 # lower right corner

        #any weird shit outside of the roi we remove as well
        frame_grayscale[:70, :] = 255
        frame_grayscale[-120:, :] = 255
        frame_grayscale[:, :180] = 255
        frame_grayscale[:, -200:] = 255

        return frame_grayscale
