from flytracker.tracker import Tracker
import numpy as np

mask = np.ones((1080, 1280), dtype=np.bool) # assumes 1080 x 1280 resolution
mask[:110, :] = 0 
mask[-110:, :] = 0 
mask[:, :180] = 0
mask[:, -260:] = 0

path = '/home/gert-jan/Documents/flyTracker/data/testing_data/4arenas/seq_1.h264'
tracker = Tracker(mask=mask, movie_path=path, output_path='tests/4arenas/')
dataset = tracker.run(n_per_batch=10**8)