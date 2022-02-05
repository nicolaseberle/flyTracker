import torch
import numpy as np
from flytracker import run
from flytracker.analysis import annotate

movie_path = "/Users/gjboth/Documents/flyTracker/data/experiments/clement/seq_1.mp4"
mapping_folder = "data/distortion_maps/"

# Making mask
mask = torch.ones((1080, 1280), dtype=bool)
mask[:160, :] = 0
mask[-120:, :] = 0
mask[:, :220] = 0
mask[:, -240:] = 0


print("Running tracker.")
df = run(movie_path, mask, n_arenas=4, gpu=False, max_change=np.inf, n_frames=5000)

# Saving
df.to_hdf("data/experiments/clement/tracks.hdf", key="df", complevel=9, complib="blosc")

output_loc = "data/experiments/bruno/videos/annotated_video.mp4"
print("Started annotating.")
# annotate(
#    df,
#    movie_path,
#    mapping_folder,
#    output_loc,
#    track_length=20,
#    touching_distance=10,
# )
