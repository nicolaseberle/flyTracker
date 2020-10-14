import cv2 as cv 
import numpy as np
from .components import Frame, BlobDetector, HungarianTracker, KMeansSKLearn, KMeansC


class FourArenasQRCodeMask:
    """Contains mask for four arenas with QR code."""
    @property
    def mask(self):
        # Building mask, this will be better later
        mask = np.ones((1080, 1280), dtype=np.bool) # assumes 1080 x 1280 resolution
        mask[:180, :300] = 0 # upper left corner
        mask[-230:, :300] = 0 # lower left corner
        mask[-230:, -300:] = 0 # lower right corner
        mask[:180, -300:] = 0 # lower right corner
        mask[:70, :] = 0
        mask[-120:, :] = 0
        mask[:, :180] = 0
        mask[:, -200:] = 0
        
        return mask


def run_tracker_C(n_frames, n_flies=40, path=None, mask=None):
    """Runs code up till first n_frames"""
    if path is None:
        path = '/Users/gert-janboth/Documents/flyTracker/data/movies/4arenas_QR.h264'
    if mask is None:
        mask = FourArenasQRCodeMask().mask
        
    capture = cv.VideoCapture(path)

    locations = []
    identities = [np.arange(n_flies)]
    
    localise_flies = BlobDetector()
    error_correct = KMeansC(n_flies)
    tracker = HungarianTracker()
    time = []

    for frame_idx in np.arange(n_frames):
        # Load frame and localize flies
        time.append(np.ones((n_flies, 1)) * frame_idx)
        frame = Frame(capture.read()[1], mask)
        fly_locations = localise_flies(frame)

        # Correct error
        if fly_locations.shape[0] != n_flies:
            fly_locations = error_correct(frame, locations[-1])
        locations.append(fly_locations)

        # Track flies
        if frame_idx > 0:
            new_frame_identities = tracker(locations[-2], locations[-1])
            locations[-1] = locations[-1][new_frame_identities, :]
            identities.append(np.arange(n_flies))

    dataset = np.concatenate((np.concatenate(time, axis=0), np.concatenate(identities, axis=0)[:, None], np.concatenate(locations, axis=0)), axis=1)
    return dataset

def run_tracker_python(n_frames, n_flies=40, path=None, mask=None):
    """Runs code up till first n_frames"""
    if path is None:
        path = '/Users/gert-janboth/Documents/flyTracker/data/movies/4arenas_QR.h264'
    if mask is None:
        mask = FourArenasQRCodeMask().mask
        
    capture = cv.VideoCapture(path)

    locations = []
    identities = [np.arange(n_flies)]
    
    localise_flies = BlobDetector()
    error_correct = KMeansSKLearn(n_flies)
    tracker = HungarianTracker()
    time = []

    for frame_idx in np.arange(n_frames):
        # Load frame and localize flies
        time.append(np.ones((n_flies, 1)) * frame_idx)
        frame = Frame(capture.read()[1], mask)
        fly_locations = localise_flies(frame)

        # Correct error
        if fly_locations.shape[0] != n_flies:
            fly_locations = error_correct(frame, locations[-1])
        locations.append(fly_locations)

        # Track flies
        if frame_idx > 0:
            new_frame_identities = tracker(locations[-2], locations[-1])
            locations[-1] = locations[-1][new_frame_identities, :]
            identities.append(np.arange(n_flies))

    dataset = np.concatenate((np.concatenate(time, axis=0), np.concatenate(identities, axis=0)[:, None], np.concatenate(locations, axis=0)), axis=1)
    return dataset


def run_localization(n_frames, n_flies=40, path=None, mask=None):
    """Runs localization up till first n_frames"""
    if path is None:
        path = '/Users/gert-janboth/Documents/flyTracker/data/movies/4arenas_QR.h264'
    if mask is None:
        mask = FourArenasQRCodeMask().mask
        
    capture = cv.VideoCapture(path)

    locations = []
    localise_flies = BlobDetector()
    error_correct = KMeansCorrect(n_flies)

    for frame_idx in np.arange(n_frames):
        # Load frame and localize flies
        frame = Frame(capture.read()[1], mask)
        fly_locations = localise_flies(frame)

        # Correct error
        if fly_locations.shape[0] != n_flies:
            fly_locations = error_correct(frame)
        locations.append(fly_locations)

    return locations


def run_tracker_kmeansCV(n_frames, n_flies=40, path=None, mask=None):
    """Runs code up till first n_frames"""
    if path is None:
        path = '/Users/gert-janboth/Documents/flyTracker/data/movies/4arenas_QR.h264'
    if mask is None:
        mask = FourArenasQRCodeMask().mask
        
    capture = cv.VideoCapture(path)

    locations = []
    identities = [np.arange(n_flies)]
    error_correct = KMeansC(n_flies)
    tracker = HungarianTracker()
    time = []

    previous_labels = None

    for frame_idx in np.arange(n_frames):
        time.append(np.ones((n_flies, 1)) * frame_idx)
        frame = Frame(capture.read()[1], mask)
        fly_locations, previous_labels = error_correct(frame, previous_labels)
        locations.append(fly_locations)

        # Track flies
        if frame_idx > 0:
            new_frame_identities = tracker(locations[-2], locations[-1])
            locations[-1] = locations[-1][new_frame_identities, :]
            identities.append(np.arange(n_flies))

    dataset = np.concatenate((np.concatenate(time, axis=0), np.concatenate(identities, axis=0)[:, None], np.concatenate(locations, axis=0)), axis=1)
    return dataset


def run_tracker_kmeanspython(n_frames, n_flies=40, path=None, mask=None):
    """Runs code up till first n_frames"""
    if path is None:
        path = '/Users/gert-janboth/Documents/flyTracker/data/movies/4arenas_QR.h264'
    if mask is None:
        mask = FourArenasQRCodeMask().mask
        
    capture = cv.VideoCapture(path)

    locations = []
    identities = [np.arange(n_flies)]
    error_correct = KMeansSKLearn(n_flies)
    tracker = HungarianTracker()
    time = []

    fly_locations = None

    for frame_idx in np.arange(n_frames):
        time.append(np.ones((n_flies, 1)) * frame_idx)
        frame = Frame(capture.read()[1], mask)
        fly_locations = error_correct(frame, fly_locations)
        locations.append(fly_locations)

        # Track flies
        if frame_idx > 0:
            new_frame_identities = tracker(locations[-2], locations[-1])
            locations[-1] = locations[-1][new_frame_identities, :]
            identities.append(np.arange(n_flies))

    dataset = np.concatenate((np.concatenate(time, axis=0), np.concatenate(identities, axis=0)[:, None], np.concatenate(locations, axis=0)), axis=1)
    return dataset