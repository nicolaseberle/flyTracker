# %% Imports
import numpy as np
import cv2
from flytracker.annotating import (
    parse_data,
    setup_loader,
    setup_writer,
    add_frame_info,
    write_ID,
    write_tracks,
)

from itertools import count
import cProfile
import pstats
from seaborn import color_palette

pr = cProfile.Profile()
pr.enable()

# %% Settings
movie_loc = "data/testing_data/bruno/seq_1.mp4"
output_loc = "notebooks/annotated_video.mp4"
df_loc = "tests/bruno/df_new.hdf"
mapping_folder = "data/distortion_maps/"
touching_distance = 12  # in pixels
track_length = 5  # in seconds


# %% Setting up
data = parse_data(df_loc)
initial_frame = data[0, 0, 0]
# plus 1 for intiial frame since we plot (n-1, n)
loader, image_size = setup_loader(
    movie_loc, mapping_folder, initial_frame=(initial_frame + 1)
)
writer = setup_writer(output_loc, image_size, fps=30)
mask = np.zeros((*image_size[::-1], 3), dtype=np.uint8)  # TODO: Check different shapes


max_frames = 1000
length = int(np.around(track_length * 30))

palette = color_palette("Paired")
color_fn = lambda idx: tuple(color * 255 for color in palette[idx % len(palette)])

# %%
for frame in count(start=1):
    lower_frame, upper_frame = np.maximum(frame - length, 0), frame
    image = loader()
    if (image is None) or (frame == (max_frames + 1)):
        break  # we're finished

    image = add_frame_info(image, f"frame: {upper_frame}")
    image = write_ID(image, data[upper_frame], touching_distance=touching_distance)
    image = write_tracks(image, data[lower_frame:upper_frame], color_fn)
    writer.write(image)
writer.release()

pr.disable()
sortby = pstats.SortKey.CUMULATIVE
ps = pstats.Stats(pr).sort_stats(sortby)
ps.print_stats(20)
