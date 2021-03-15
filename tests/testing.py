import numpy as np
from torch.utils.data import DataLoader

from flytracker.tracker import _run
from flytracker.analysis import annotate
from flytracker.io import VideoDataset
from flytracker.preprocessing import preprocessing
from flytracker.localization.blob import localize_blob
from flytracker.localization.kmeans import localize_kmeans_torch, localize_kmeans
from flytracker.tracking import tracking
from flytracker.analysis import post_process

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

print("Running tracker.")


dataset = VideoDataset(movie_path, preprocessing, mask, mapping_folder)
loader = DataLoader(dataset, batch_size=1, pin_memory=True)
df = _run(
    loader,
    localize_blob,
    localize_kmeans_torch,
    (120, "cuda", 1e-4),
    tracking,
    post_process,
    n_arenas=4,
    n_frames=1000,
    n_ini=100,
)

df.to_hdf("tests/df.hdf", key="df", complevel=9, complib="blosc")


# print("Starting annotating.")
# annotate(
#    df,
#    movie_path,
#    mapping_folder,
#    "tests/annotated_video.mp4",
#    max_frames=1000,
#    track_length=30,
#    touching_distance=10,
# )
