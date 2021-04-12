import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from itertools import takewhile
from typing import Callable, Iterable, Tuple

from .io import VideoDataset
from .preprocessing import preprocessing
from .localization.blob import localize_blob, default_blob_detector_params
from .localization.kmeans import localize_kmeans, localize_kmeans_torch
from .tracking import tracking
from .analysis import post_process


def run(
    movie_path: str,
    mask: np.ndarray,
    n_arenas: int,
    mapping_folder: str,
    n_frames: int = np.inf,
    n_ini: int = 100,
    gpu: bool = True,
) -> pd.DataFrame:
    """User facing run function with sensible standard settings."""

    dataset = VideoDataset(movie_path, preprocessing, mask, mapping_folder)
    loader = DataLoader(dataset, batch_size=1, pin_memory=True)
    if gpu:
        main_localizer = localize_kmeans_torch
        localizer_args = (120, "cuda", 1e-4)
    else:
        main_localizer = localize_kmeans
        localizer_args = (120, 1e-4)
    blob_args = (default_blob_detector_params(),)
    return _run(
        loader,
        localize_blob,
        blob_args,
        main_localizer,
        localizer_args,
        tracking,
        post_process,
        n_arenas,
        n_frames,
        n_ini,
    )


def _run(
    loader: Iterable,
    initial_localizer: Callable,
    initial_localizer_args: Tuple,
    main_localizer: Callable,
    main_localizer_args: Tuple,
    tracker: Callable,
    post_process: Callable,
    n_arenas: int,
    n_frames=np.inf,
    n_ini=100,
):
    initial_position, initial_frame = _initialize(
        loader, initial_localizer, initial_localizer_args, n_ini
    )
    locations = _localize(
        loader, main_localizer, main_localizer_args, initial_position, n_frames
    )
    ordered_locations = tracker(locations)
    df = post_process(ordered_locations, initial_frame, n_arenas)
    return df


def _initialize(
    loader: Iterable,
    preprocessor: Callable,
    localizer: Callable,
    localizer_args,
    n_frames: int,
) -> Tuple[np.ndarray, int]:
    n_blobs = []
    for frame_idx, image in enumerate(loader):
        image = preprocessor(image).numpy().squeeze()
        locations = localizer(image, *localizer_args)
        n_blobs.append(locations.shape[0])

        if frame_idx >= n_frames:
            n_flies = int(np.median(n_blobs))
            if n_blobs[-1] == n_flies:
                break

    return locations, frame_idx


def _localize(
    loader: Iterable,
    localizer: Callable,
    localizer_args: Tuple,
    initial_position: np.ndarray,
    n_frames: int,
) -> np.ndarray:

    locations = [initial_position]
    for frame_idx, image in takewhile(lambda x: x[0] <= n_frames, enumerate(loader)):
        locations = localizer(image, locations, *localizer_args)
        if frame_idx % 1000 == 0:
            print(f"Done with frame {frame_idx}")

    return locations
