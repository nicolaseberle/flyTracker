# %% Imports
import numpy as np
import cv2
from annotating_code import *
from itertools import count
import pandas as pd


# %% Settings
movie_loc = "/home/gert-jan/Documents/flyTracker/data/testing_data/4arenas/seq_1.mp4"
output_loc = "/home/gert-jan/Documents/flyTracker/tests/4arenas/annotated_video.avi"
df_loc = "/home/gert-jan/Documents/flyTracker/tests/4arenas/df_batch_0.hdf"
mapping_folder = "/home/gert-jan/Documents/flyTracker/data/distortion_maps/"


# %% Setup
df = pd.read_hdf(df_loc, key="df")
# this is important; we expect a sorted daatframe
df = df.sort_values(by=["frame", "ID"])

mask = None
loader, image_size = setup_loader(
    movie_loc, mapping_folder, initial_frame=(df.frame.min() + 1)
)
writer = setup_writer(output_loc, image_size, fps=30)

n_minus_one_iterator = df.groupby("frame")  # gives data of frame n-1
n_iterator = df.query(f"frame > {df.frame.min()}").groupby(
    "frame"
)  # gives data of frame n

# %% Running
max_frames = 500
for (idx, df_n_minus_one), (_, df_n) in zip(n_minus_one_iterator, n_iterator):
    image = loader()
    if image is None:
        break  # we're finished

    if mask is None:
        mask = np.zeros_like(image)

    image = add_frame_info(image, f"frame: {df_n.frame.iloc[0]}")
    image = write_ID(image, df_n)

    mask = update_mask(mask, df_n_minus_one, df_n)
    new_image = image * (np.sum(mask, axis=-1) == 0)[:, :, None] + mask
    mask = cv2.addWeighted(
        mask, 0.99, image * (np.sum(mask, axis=-1) != 0)[:, :, None], 0.01, -5
    )

    writer.write(new_image)

    if idx == max_frames:
        break
writer.release()


# %%

# %%
