import numpy as np
from flytracker.tracker import run

movie_path = "data/experiments/bruno/videos/seq_1.mp4"
mapping_folder = "data/distortion_maps/"


mask = np.ones((1080, 1280), dtype=bool)
mask[:160, :] = 0
mask[-170:, :] = 0
mask[:, :300] = 0
mask[:, -230:] = 0

mask[:220, :400] = 0
mask[:230, -300:] = 0
mask[-250:, :370] = 0
mask[830:, 970:] = 0

df = run(movie_path, mask, n_arenas=4, mapping_folder=mapping_folder, n_frames=1000)
df.to_hdf("tests/df.hdf", key="df", complevel=9, complib="blosc")
