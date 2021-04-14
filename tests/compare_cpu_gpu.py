import torch
import numpy as np
from flytracker.io import DataLoader

from flytracker.tracker import _initialize, _localize
from flytracker.preprocessing.preprocessing import (
    preprocessing_kmeans,
    preprocessing_blob,
)
from flytracker.localization.blob import localize_blob, default_blob_detector_params
from flytracker.localization.kmeans import (
    localize_kmeans_torch,
    localize_kmeans_sklearn,
)
from time import time

movie_path = "data/experiments/bruno/videos/seq_1.mp4"
n_frames = 5000

mask = torch.ones((1080, 1280), dtype=bool)
mask[:130, :] = 0
mask[-160:, :] = 0
mask[:, :270] = 0
mask[:, -205:] = 0

mask[:190, :350] = 0
mask[:195, -270:] = 0
mask[-220:, :340] = 0
mask[870:, 1010:] = 0


# Running parallel sklearn
# parallel laoding doenst work for cpu
start = time()
loader = DataLoader(movie_path, parallel=False)
preprocessor_ini = preprocessing_blob(mask)
localize_ini = localize_blob(default_blob_detector_params())
initial_position, initial_frame = _initialize(
    loader, preprocessor_ini, localize_ini, 100
)

preprocessor_main = preprocessing_kmeans(mask, device="cpu")
localize_main = localize_kmeans_sklearn(120, 1e-4)
locs_cpu = _localize(
    loader, preprocessor_main, localize_main, initial_position, n_frames, "cpu"
)
loader.stop()
stop = time()
print(f"Time for CPU run: {stop - start}s")

# Running parallel gpu
start = time()
loader = DataLoader(movie_path, parallel=True)
preprocessor_ini = preprocessing_blob(mask)
localize_ini = localize_blob(default_blob_detector_params())
initial_position, initial_frame = _initialize(
    loader, preprocessor_ini, localize_ini, 100
)

preprocessor_main = preprocessing_kmeans(mask, device="cuda")
localize_main = localize_kmeans_torch(120, 1e-4, "cuda")
locs_gpu = _localize(
    loader, preprocessor_main, localize_main, initial_position, n_frames, "cuda"
)
loader.stop()
stop = time()
print(f"Time for GPU run: {stop - start}s")
print(
    f"Parallel and sequential give same result: {np.max(np.stack(locs_cpu) - torch.stack(locs_gpu).cpu().numpy())}"
)

