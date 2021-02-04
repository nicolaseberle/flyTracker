# %% Imports
import numpy as np
from flytracker.annotating import (
    parse_data,
    setup_loader,
    setup_writer,
    add_frame_info,
    write_ID,
    write_tracks,
    color_picker,
)

from itertools import count
import os

# %% Settings
movie_loc = "data/testing_data/bruno/seq_1.mp4"
output_loc = "tests/bruno/annotated_video.mp4"
df_loc = "tests/bruno/df_new.hdf"
mapping_folder = "data/distortion_maps/"

touching_distance = 10  # in pixels
track_length = 4  # in seconds


# %% Setting up
data, n_flies_per_arena = parse_data(df_loc)
initial_frame = data[0, 0, 0]
# plus 1 for intiial frame since we plot (n-1, n)
loader, image_size = setup_loader(
    movie_loc, mapping_folder, initial_frame=(initial_frame + 1)
)
writer = setup_writer(output_loc, image_size, fps=30)
mask = np.zeros((*image_size[::-1], 3), dtype=np.uint8)  # TODO: Check different shapes


max_frames = 10 ** 9
length = int(np.around(track_length * 30))
color_fn = lambda ID: color_picker(ID, n_flies_per_arena)
# %%
for frame in count(start=1):
    lower_frame, upper_frame = np.maximum(frame - length, 0), frame
    image = loader()
    if (image is None) or (frame == (max_frames + 1)):
        break  # we're finished

    image = add_frame_info(image, f"frame: {upper_frame}")
    # First write tracks so that numbers don't get occluded.
    image = write_tracks(image, data[lower_frame:upper_frame], color_fn)
    image = write_ID(image, data[upper_frame], touching_distance=touching_distance)
    writer.write(image)

    if frame % 1000 == 0:
        print(f"Done with frame {frame}")
writer.release()

# Compressing to h264 with ffmpeg
compressed_loc = output_loc.split(".")[0] + "_compressed.mp4"
os.system(f"ffmpeg -i {output_loc} -an -vcodec libx264 -crf 23 {compressed_loc}")
