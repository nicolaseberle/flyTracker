import numpy as np
from flytracker import run
from flytracker.analysis import annotate

movie_path = "data/experiments/bruno/videos/seq_1.mp4"
mapping_folder = "data/distortion_maps/"

# Making mask
mask = np.ones((1080, 1280), dtype=bool)
mask[:160, :] = 0
mask[-170:, :] = 0
mask[:, :300] = 0
mask[:, -230:] = 0

mask[:220, :400] = 0
mask[:230, -300:] = 0
mask[-250:, :370] = 0
mask[830:, 970:] = 0

print("Running tracker.")
df = run(
    movie_path,
    mask,
    n_arenas=4,
    # mapping_folder=mapping_folder,
    gpu=False,
)

# Saving
df.to_hdf(
    "data/experiments/bruno/results/df.hdf", key="df", complevel=9, complib="blosc"
)

output_loc = "data/experiments/bruno/videos/annotated_video.mp4"
print("Started annotating.")
annotate(
    df,
    movie_path,
    mapping_folder,
    output_loc,
    track_length=20,
    touching_distance=10,
)
