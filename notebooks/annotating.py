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
loader, image_size = setup_loader(movie_loc, mapping_folder)
writer = setup_writer(output_loc, image_size, fps=30)


df = pd.read_hdf(df_loc, key="df")
df = df.sort_values(by=["frame", "ID"])

mask = None

# %% Running
for frame_idx in np.arange(400):  # count():
    image = loader()
    if image is None:
        break  # we're finished

    if mask is None:
        mask = np.zeros_like(image)

    image = add_frame_info(image, f"frame: {frame_idx}")
    image = touching(image, df, frame_idx, touching_distance=15)

    mask = update_mask(mask, df, frame_idx)
    new_image = image * (np.sum(mask, axis=-1) == 0)[:, :, None] + mask
    mask = cv2.addWeighted(
        mask, 0.99, image * (np.sum(mask, axis=-1) != 0)[:, :, None], 0.01, -5
    )

    writer.write(new_image)
writer.release()


# %%

# %%
