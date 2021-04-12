import torch
from flytracker import run
from flytracker.analysis import annotate
from time import time

movie_path = "data/experiments/bruno/videos/seq_1.mp4"

mask = torch.ones((1080, 1280), dtype=bool)
mask[:130, :] = 0
mask[-160:, :] = 0
mask[:, :270] = 0
mask[:, -205:] = 0

mask[:190, :350] = 0
mask[:195, -270:] = 0
mask[-220:, :340] = 0
mask[870:, 1010:] = 0

print("Running tracker.")
start = time()
df = run(
    movie_path,
    mask,
    n_arenas=4,
    n_frames=5000,
    gpu=True,
    parallel=True,
    n_ini=100,
    threshold=120,
)
stop = time()
print(f"Running took {stop-start}s")
# df.to_hdf("tests/df.hdf", key="df", complevel=9, complib="blosc")

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
