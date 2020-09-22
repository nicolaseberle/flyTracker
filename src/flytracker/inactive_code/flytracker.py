import cv2 as cv 
import numpy as np


class FlyTracker:
    '''This class implements the flytracker pipeline.'''
    def __init__(self, preprocessing, localizing):
        self.preprocessing = preprocessing
        self.localizing = localizing


    def run_pipeline(self, video_path, output_folder, max_frames=1000):
        #assert video_path.split('.')[-1] == 'mp4', 'Only mp4 is supported.'

        capture = cv.VideoCapture(video_path)

        locations = []
        frame_read = True
        frame_number = 0
        
        while (frame_read) and (frame_number < max_frames):
            _, frame = capture.read()
            preprocessed_frame = self.preprocessing(frame)
            locations.append(self.localizing(preprocessed_frame, frame))
            frame_number += 1
        return locations

