# %% Imports
import numpy as np
import cv2
import pandas as pd
from os.path import join

from pandas.core.algorithms import value_counts

# %% Locations
movie_loc = "/home/gert-jan/Documents/flyTracker/data/testing_data/4arenas/seq_1.mp4"
output_loc = "/home/gert-jan/Documents/flyTracker/tests/4arenas/annotated_video.avi"
df_loc = "/home/gert-jan/Documents/flyTracker/tests/4arenas/df_batch_0.hdf"
mapping_folder = "/home/gert-jan/Documents/flyTracker/data/distortion_maps/"


# Setting up loader and writer.
cap = cv2.VideoCapture(movie_loc)

fps, height, width = (
    30,
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(output_loc, fourcc, fps, (width, height), False,)

df = pd.read_hdf(df_loc, key="df")
df = df.sort_values(by=["frame", "ID"])

# %% Function which adds frame info (e.g. frame), to upper left corner)
add_frame_info = lambda img, text: cv2.putText(
    img,
    text,
    org=(50, 50),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=1,
    color=(0, 0, 0),
    thickness=2,
)


def construct_undistort_map(image_size, folder):
    """ Construct openCV undistort undistort mapping. Make sure files as named below are the supplied folder.
    Returns a function which takes in image and returns the undistorted image."""
    mtx = np.load(join(folder, "mtx_file.npy"))
    dist = np.load(join(folder, "dist_file.npy"))
    newcameramtx = np.load(join(folder, "newcameramtx_file.npy"))

    mapping = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, image_size, 5)
    return mapping


mapping = construct_undistort_map((width, height), mapping_folder)


# %% function which draw line for a single fly
length = 100

# %% Writing
for frame_idx in np.arange(300):
    frame = cap.read()[1]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.remap(img, *mapping, cv2.INTER_LINEAR)

    img = add_frame_info(img, f"Frame {frame_idx}")
    pts = [
        df.query(f"{frame_idx - length} < frame <={frame_idx} and ID == {ID}")[
            ["x", "y"]
        ].to_numpy(dtype=np.int32)[:, None, :]
        for ID in np.arange(40)
    ]

    if pts[0].shape[0] > 1:
        img = cv2.polylines(img, pts, False, (0, 0, 0))
    writer.write(img)
cap.release()
writer.release()

# %%

# %%

frame_idx = 150
local_df = df.query(f"{frame_idx - length} < frame <={frame_idx}")
pivoted_df = pd.pivot_table(local_df, index="frame", columns="ID", values=["x", "y"])
# %%
