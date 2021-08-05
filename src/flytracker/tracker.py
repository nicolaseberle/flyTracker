from operator import pos
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
import os
from natsort import natsorted


def run(
    path: str,
    mask: torch.Tensor,
    n_arenas: int,
    n_frames: int = np.inf,
    n_ini: int = 100,
    gpu: bool = True,
    threshold: int = 120,
    max_change: float = 20.0,
) -> pd.DataFrame:
    """User facing run function with sensible standard settings."""

    # Setting up GPU/CPU settings
    if gpu:
        device = "cuda"
        main_localizer = localize_kmeans_torch
        localizer_args = (
            torch.tensor(threshold).to(device, non_blocking=True),
            1e-4,
            device,
        )
    else:
        device = "cpu"
        main_localizer = localize_kmeans_sklearn
        localizer_args = (torch.tensor(threshold), 1e-4)

    # Running
    return _run(
        loaders(path),
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
        max_change,
    )


def _initialize(
    loader: DataLoader, preprocessor: Callable, localizer: Callable, n_init_frames: int,
):
    """ Finds an initial location of the flies and the number of flies using the given localizer."""
    n_blobs = []
    for enum_idx, (frame_idx, image) in enumerate(loader):
        locations = localizer(preprocessor(image))
        n_blobs.append(locations.shape[0])

        if enum_idx >= n_init_frames:
            n_flies = int(np.median(n_blobs))
            if n_blobs[-1] == n_flies:
                break
    return locations


def initialize_pos_array(n_frames, init, device, init_frame=0):
    """ Constructs empty array to be filled with positions. Init_frame sets which frame should be used for init. infers shapes from init.
    Return shsape (n_frames, init.shape[0], init.shape[1])."""
    pos_array = torch.zeros((n_frames, init.shape[0], init.shape[1]), device=device)
    pos_array[init_frame] = torch.tensor(init, dtype=torch.float32)
    return pos_array


def loaders(path):
    """ Construct iterable of dataloaders; contains either all videos in a folder (if path is folder)
    or just the single video (if path is a specific video."""
    if os.path.isfile(path) and path[-4:] == ".mp4":
        files = [path]
    elif os.path.isdir(path):
        files = [
            os.path.join(path, file)
            for file in filter(lambda file: file[-4:] == ".mp4", os.listdir(path))
        ]
    else:
        raise NotADirectoryError

    return [DataLoader(path) for path in natsorted(files)]


def _localize(
    loader: DataLoader,
    preprocessor: Callable,
    localizer: Callable,
    init: torch.Tensor,
    n_frames: int,
    device: str,
    max_change: float,
):
    """ Localize flies in single video. Init sets initial guess, skips frames where the dist between two frames > max_change."""
    initializing_frame = np.maximum(loader.current_frame - 1, 0)
    locations = initialize_pos_array(
        loader.frames, init, device, init_frame=initializing_frame
    )
    for _, (frame_idx, image) in takewhile(
        lambda x: x[0] <= n_frames, enumerate(loader)
    ):
        image = image.to(device, non_blocking=True)
        new_positions, delta_position = localizer(
            preprocessor(image), locations[initializing_frame]
        )

        if delta_position < max_change:
            locations[frame_idx] = new_positions
            initializing_frame = frame_idx

        if frame_idx % 1000 == 0:
            print(f"Done with frame {frame_idx}")
    loader.stop()
    return locations


def _run(
    loaders: Iterable[DataLoader],
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
    max_change: float,
):
    """Runs the tracker over all provided videos and postprocesses etc. Returns dataframe with all data."""

    # initialize and set up
    init = _initialize(loaders[0], initial_preprocessor, initial_localizer, n_ini)
    positions = [[init]]
    localizer = lambda loader, init: _localize(
        loader, main_preprocessor, main_localizer, init, n_frames, device, max_change,
    )

    # Run actual tracker
    for loader in loaders:
        positions.append(localizer(loader, positions[-1][-1]))

    # Postprocessing
    positions = torch.cat(positions[1:], axis=0).cpu()  # removing init
    non_zero_frames = torch.sum(positions, axis=[1, 2]) != 0
    time = (
        torch.arange(positions.shape[0])
        .repeat(repeats=(positions.shape[1], 1))
        .T[..., None]
    )
    positions = torch.cat((positions, time), axis=-1)
    positions = tracker(positions[non_zero_frames])
    df = post_process(positions, n_arenas)
    return df
