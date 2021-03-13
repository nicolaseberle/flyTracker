import cv2 as cv
from ..preprocessing.preprocessing import preprocessing


def opencv_loader(movie_path, mapping_folder, mask):
    capture = cv.VideoCapture(movie_path)
    preprocessor = preprocessing(mask, mapping_folder)
    return lambda: preprocessor(capture.read()[1])
