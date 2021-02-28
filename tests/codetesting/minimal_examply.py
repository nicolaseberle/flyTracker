import numpy as np
import cv2 as cv
from flytracker.preprocessing import construct_undistort_map, preprocessing
from flytracker.tracker import run
import matplotlib.pyplot as plt

# %% Mask
movie_path = "/home/gert-jan/Documents/flyTracker/data/testing_data/4arenas/seq_1.mp4"
mapping_folder = "/home/gert-jan/Documents/flyTracker/data/distortion_maps/"

# capture = cv.VideoCapture(movie_path)
# image_size = (
#    int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
#    int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)),
# )
# mapping = construct_undistort_map(image_size, mapping_folder)
# image = lambda mask: preprocessing(capture.read()[1], mapping=mapping, mask=mask)

# Creating mask
mask = np.ones((1080, 1280), dtype=np.bool)  # assumes 1080 x 1280 resolution
mask[:110, :] = 0
mask[-110:, :] = 0
mask[:, :180] = 0
mask[:, -260:] = 0


# plt.figure(figsize=(10, 10))
# plt.imshow(image(mask), cmap="gray")

# %% Running
df = run(movie_path, mask, n_arenas=4, n_frames=1000)

df.to_hdf(
    "/home/gert-jan/Documents/flyTracker/tests/codetesting/df.hdf",
    key="df",
    complevel=9,
    complib="blosc",
)

