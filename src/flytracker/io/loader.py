import cv2 as cv
from ..preprocessing.undistort_mapping import construct_undistort_map
from ..preprocessing.preprocessing import preprocessing


def opencv_loader(movie_path, mapping_folder, mask):
    capture = cv.VideoCapture(movie_path)

    mapping = construct_undistort_map(mask.shape[::-1], mapping_folder)
    loader = lambda: preprocessing(capture.read()[1], mapping=mapping, mask=mask)
    return loader
