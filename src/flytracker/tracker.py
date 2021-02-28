import numpy as np
import pandas as pd

from flytracker.dataset import VideoDataset
from torch.utils.data import DataLoader
from .postprocessing import post_process
from .tracking import (
    blob_detector_localization,
    localize_kmeans,
    localize_kmeans_jax,
    localize_kmeans_torch,
    hungarian,
)
import torch


def run(
    movie_path: str,
    mask: np.ndarray,
    n_arenas: int,
    n_frames: int = None,
    n_ini: int = 100,
) -> pd.DataFrame:
    """Runs the whole pipeline. I.e preprocessing, localizing and postprocessing."""

    # Constructing loader
    dataset = VideoDataset(movie_path, mask)
    loader = DataLoader(dataset, batch_size=1, pin_memory=True)

    # Actual logic
    n_flies, initial_locations, initial_frame = initialize(loader, n_ini)
    locs = localize_kmeans_torch(loader, initial_locations, n_frames=n_frames)
    df = post_process(locs, initial_frame, n_arenas=n_arenas)
    return df


def initialize(loader, n_frames=100):
    """Find flies using blob detector and
    calulate number of flies."""
    n_blobs = []

    for frame_idx, frame in enumerate(loader):
        locations = blob_detector_localization(frame.numpy().squeeze())
        n_blobs.append(locations.shape[0])
        if len(n_blobs) >= n_frames:
            n_flies = int(np.median(n_blobs))
            if n_blobs[-1] == n_flies:
                break
    # pluse on cause the next one is the first
    initial_frame = frame_idx + 1
    locations = torch.tensor(locations[:, [-1, 0]], dtype=torch.float32).to("cuda")
    return n_flies, locations, initial_frame

