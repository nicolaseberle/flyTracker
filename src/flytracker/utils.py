import cv2 as cv 
import numpy as np
from .components import Frame, BlobDetector, Tracker


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


def run_tracker(n_frames, n_flies=40, path=None, mask=None):
    """Runs code up till first n_frames"""
    if path is None:
        path = '/Users/gert-janboth/Documents/flyTracker/data/movies/4arenas_QR.h264'
    if mask is None:
        mask = FourArenasQRCodeMask().mask
        
    capture = cv.VideoCapture(path)

    locations = []
    identities = [np.arange(n_flies)[:, None]]
    
    localise_flies = BlobDetector()
    tracker = Tracker()
    time = []

    for frame_idx in np.arange(n_frames):
        time.append(np.ones((n_flies, 1)) * frame_idx)
        frame = Frame(capture.read()[1], mask)
        locations.append(localise_flies(frame))
        if frame_idx > 0:
            identities.append(tracker(locations[-2], locations[-1], identities[-1].squeeze()))
    dataset = np.concatenate((np.concatenate(time, axis=0), np.concatenate(identities, axis=0), np.concatenate(locations, axis=0)), axis=1)
    return dataset
