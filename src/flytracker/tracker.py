import numpy as np
import pandas as pd
import torch
from itertools import takewhile
from typing import Callable, Iterable

from .io import DataLoader
from .preprocessing import preprocessing_blob, preprocessing_kmeans
from .localization.blob import localize_blob, default_blob_detector_params
from .localization.kmeans import localize_kmeans_sklearn, localize_kmeans_torch
from .tracking import tracking
from .analysis import post_process


def run(
    movie_path: str,
    mask: torch.Tensor,
    n_arenas: int,
    n_frames: int = np.inf,
    n_ini: int = 100,
    gpu: bool = True,
    parallel: bool = True,
    threshold: int = 120,
) -> pd.DataFrame:
    """User facing run function with sensible standard settings."""

    if gpu:
        device = "cuda"
        main_localizer = localize_kmeans_torch
        localizer_args = (threshold, 1e-4, device)
    else:
        device = "cpu"
        parallel = False  # parallel doesnt work on cpu
        main_localizer = localize_kmeans_sklearn
        localizer_args = (threshold, 1e-4)

    loader = DataLoader(movie_path, parallel=parallel, max_queue=20)
    return _run(
        loader,
        preprocessing_blob(mask),
        localize_blob(default_blob_detector_params()),
        preprocessing_kmeans(mask, device=device),
        main_localizer(*localizer_args),
        tracking,
        post_process,
        n_arenas,
        n_frames,
        n_ini,
        device,
    )


def _run(
    loader: Iterable,
    initial_preprocessor: Callable,
    initial_localizer: Callable,
    main_preprocessor: Callable,
    main_localizer: Callable,
    tracker: Callable,
    post_process: Callable,
    n_arenas: int,
    n_frames: int,
    n_ini: int,
    device: str,
):

    initial_position, initial_frame = _initialize(
        loader, initial_preprocessor, initial_localizer, n_ini,
    )

    locations = _localize(
        loader, main_preprocessor, main_localizer, initial_position, n_frames, device,
    )
    loader.stop()
    ordered_locations = tracker(locations)
    df = post_process(ordered_locations, initial_frame, n_arenas)
    return df


def _initialize(
    loader: Iterable, preprocessor: Callable, localizer: Callable, n_frames: int,
):
    n_blobs = []
    for frame_idx, image in enumerate(loader):
        locations = localizer(preprocessor(image))
        n_blobs.append(locations.shape[0])

        if frame_idx >= n_frames:
            n_flies = int(np.median(n_blobs))
            if n_blobs[-1] == n_flies:
                break

    return torch.tensor(locations, dtype=torch.float32), frame_idx


def _localize(
    loader: Iterable,
    preprocessor: Callable,
    localizer: Callable,
    initial_position: torch.Tensor,
    n_frames: int,
    device: str,
):

    locations = [initial_position.to(device, non_blocking=True)]
    for frame_idx, image in takewhile(lambda x: x[0] <= n_frames, enumerate(loader)):
        image = image.to(device, non_blocking=True)
        frame_locs = localizer(preprocessor(image), locations[-1])
        locations.append(frame_locs)
        if frame_idx % 1000 == 0:
            print(f"Done with frame {frame_idx}")

    return locations
