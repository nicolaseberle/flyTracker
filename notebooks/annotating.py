# %% Imports
import numpy as np
import cv2
from flytracker.annotating import (
    parse_data,
    setup_loader,
    setup_writer,
    add_frame_info,
    write_ID,
    update_mask,
)

from itertools import count
import cProfile
import pstats

pr = cProfile.Profile()
pr.enable()

# %% Settings
movie_loc = "data/testing_data/bruno/seq_1.mp4"
output_loc = "notebooks/annotated_video.mp4"
df_loc = "tests/bruno/df_new.hdf"
mapping_folder = "data/distortion_maps/"
touching_distance = 12


data = parse_data(df_loc)
initial_frame = data[0, 0, 0]
# plus 1 for intiial frame since we plot (n-1, n)
loader, image_size = setup_loader(
    movie_loc, mapping_folder, initial_frame=(initial_frame + 1)
)
writer = setup_writer(output_loc, image_size, fps=30)
mask = np.zeros((*image_size[::-1], 3), dtype=np.uint8)  # TODO: Check different shapes


max_frames = 1000
length = 1

for frame in count(start=1):
    lower_frame, upper_frame = np.maximum(frame - length, 0), frame
    image = loader()
    if (image is None) or (frame == (max_frames + 1)):
        break  # we're finished

    image = add_frame_info(image, f"frame: {upper_frame}")
    image = write_ID(image, data[upper_frame], touching_distance=touching_distance)

    mask = update_mask(mask, data[lower_frame], data[upper_frame])
    image = cv2.addWeighted(
        image, 1.0, mask, 1.0, gamma=0
    )  # image * (np.sum(mask, axis=-1) == 0)[:, :, None] + mask
    # mask = cv2.addWeighted(
    #    mask, 0.99, image * (np.sum(mask, axis=-1) != 0)[:, :, None], 0.01, -5
    # )

    writer.write(image)
writer.release()

pr.disable()
sortby = pstats.SortKey.CUMULATIVE
ps = pstats.Stats(pr).sort_stats(sortby)
ps.print_stats(20)
