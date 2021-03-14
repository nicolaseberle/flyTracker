import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from itertools import takewhile


from typing import Callable, Iterable, Tuple
from .io import VideoDataset
from .localization.blob import blob_detector_localization
from .localization.kmeans import localize_kmeans
from .analysis import post_process
from .tracking import tracking
from .preprocessing import preprocessing


def run(
    movie_path: str,
    mask: np.ndarray,
    n_arenas: int,
    mapping_folder: str,
    n_frames: int = None,
    n_ini: int = 100,
) -> pd.DataFrame:
    """User facing run function with sensible standard settings."""

    dataset = VideoDataset(movie_path, preprocessing, mask, mapping_folder)
    loader = DataLoader(dataset, batch_size=1, pin_memory=True)
    return _run(
        loader,
        blob_detector_localization,
        localize_kmeans,
        tracking,
        post_process,
        n_arenas,
        n_frames,
        n_ini,
    )


def _run(
    loader: Iterable,
    initial_localizer: Callable,
    main_localizer: Callable,
    tracker: Callable,
    post_process: Callable,
    n_arenas: int,
    n_frames=np.inf,
    n_ini=100,
):
    initial_position, initial_frame = _initialize(loader, initial_localizer, n_ini)
    locations = _localize(loader, main_localizer, initial_position, n_frames)
    ordered_locations = tracker(locations)
    df = post_process(ordered_locations, initial_frame, n_arenas)
    return df


def _initialize(
    loader: Iterable, localizer: Callable, n_frames: int
) -> Tuple[np.ndarray, int]:
    n_blobs = []
    for frame_idx, image in enumerate(loader):
        locations = localizer(image)
        n_blobs.append(locations.shape[0])

        if frame_idx >= n_frames:
            n_flies = int(np.median(n_blobs))
            if n_blobs[-1] == n_flies:
                break

    return locations, frame_idx


def _localize(
    loader: Iterable,
    localizer: Callable,
    initial_position: np.ndarray,
    n_frames: int = np.inf,
) -> np.ndarray:

    locations = [initial_position]
    for frame_idx, image in takewhile(lambda x: x[0] <= n_frames, enumerate(loader)):
        locations.append(localizer(image, locations[-1]))
        if frame_idx % 1000 == 0:
            print(f"Done with frame {frame_idx}")
    return locations
