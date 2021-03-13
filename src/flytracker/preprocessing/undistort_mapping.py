from typing import Callable, Tuple
import numpy as np
import cv2 as cv
from os.path import join


def construct_undistort_map(image_size: Tuple[int], folder: str) -> Callable:
    """ Construct openCV undistort undistort mapping. Make sure files as named below are the supplied folder.
    Returns a function which takes in image and returns the undistorted image."""
    # TODO: DOuble check image size for initdistort; are we not messing up directios?
    mtx = np.load(join(folder, "mtx_file.npy"))
    dist = np.load(join(folder, "dist_file.npy"))
    newcameramtx = np.load(join(folder, "newcameramtx_file.npy"))

    mapping = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, image_size, 5)
    return mapping
