# %% Imports
import cv2 as cv
from flytracker.preprocessing import construct_undistort_map, preprocessing
from flytracker.tracker import run
import numpy as np
import matplotlib.pyplot as plt

# %% Mask
movie_path = "/home/gert-jan/Documents/flyTracker/data/testing_data/bruno/seq_1.h264"
mapping_folder = "/home/gert-jan/Documents/flyTracker/data/distortion_maps/"

capture = cv.VideoCapture(movie_path)
image_size = (
    int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)),
)
mapping = construct_undistort_map(image_size, mapping_folder)
image = lambda mask: preprocessing(capture.read()[1], mapping=mapping, mask=mask)

# Creating mask
mask = np.ones(image_size, dtype=bool).T
mask[:160, :] = 0
mask[-170:, :] = 0
mask[:, :300] = 0
mask[:, -230:] = 0

mask[:220, :400] = 0
mask[:230, -300:] = 0
mask[-250:, :370] = 0
mask[830:, 970:] = 0


plt.figure(figsize=(10, 10))
plt.imshow(image(mask), cmap="gray")

# %% Running
df = run(movie_path, mask, n_arenas=4, mapping_folder=mapping_folder, n_frames=1000)
df.to_hdf(
    "/home/gert-jan/Documents/flyTracker/tests/codetesting/df.hdf",
    key="df",
    complevel=9,
    complib="blosc",
)
