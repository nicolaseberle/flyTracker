import cv2 as cv 
import numpy as np


class FlyTracker:
    '''This class implements the flytracker pipeline.'''
    def __init__(self, preprocessing, localizing):
        self.preprocessing = preprocessing
        self.localizing = localizing

    def run_pipeline(self, video_path, output_folder):
        assert video_path.split('.')[-1] == 'mp4', 'Only mp4 is supported.'

        capture = cv.VideoCapture(video_path)
        n_frames = capture.get(cv.CAP_PROP_FRAME_COUNT)

        locations = []
        for frame_idx in np.arange(n_frames)[:100]:
            _, frame = capture.read()
            preprocessed_frame = self.preprocessing(frame)
            locations.append(self.localizing(preprocessed_frame))
            if frame_idx % 10 == 0:
                print(f'Done with frame {frame_idx}')
        np.save(output_folder + 'locations.npy', locations)
        return np.array(locations)

