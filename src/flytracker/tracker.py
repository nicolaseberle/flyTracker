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

    # Setting up loaders
    if os.path.isfile(path) and path[-4:] == ".mp4":
        files = [path]
    elif os.path.isdir(path):
        files = [
            os.path.join(path, file)
            for file in filter(lambda file: file[-4:] == ".mp4", os.listdir(path))
        ]
    else:
        raise NotADirectoryError
    # loaders = [DataLoader(path, parallel=False) for path in natsorted(files)]

    # Running
    return _multi_run(
        files,
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


def _run(
    loader: DataLoader,
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

    positions = _initialize(
        loader, initial_preprocessor, initial_localizer, n_ini, device
    )
    positions = _localize(
        loader,
        main_preprocessor,
        main_localizer,
        positions,
        n_frames,
        device,
        max_change,
    )
    non_zero_frames = torch.sum(positions[:, :, :2], axis=[1, 2]) != 0
    positions = tracker(positions[non_zero_frames])
    df = post_process(positions, n_arenas)
    return df


def _initialize(
    loader: DataLoader,
    preprocessor: Callable,
    localizer: Callable,
    n_init_frames: int,
    device,
):

    n_blobs = []
    for enum_idx, (frame_idx, image) in enumerate(loader):
        locations = localizer(preprocessor(image))
        n_blobs.append(locations.shape[0])

        if enum_idx >= n_init_frames:
            n_flies = int(np.median(n_blobs))
            if n_blobs[-1] == n_flies:
                break
    print(enum_idx, frame_idx)
    pos_array = torch.zeros((loader.frames, n_flies, 3), device=device)
    pos_array[:, :, -1] = torch.arange(loader.frames)[:, None]  # Adding time
    pos_array[enum_idx, :, :2] = torch.tensor(locations, dtype=torch.float32)
    return pos_array


def _multi_initialize(
    loader: DataLoader, init, device,
):
    n_flies = init.shape[0]
    pos_array = torch.zeros((loader.frames, n_flies, 3), device=device)
    pos_array[:, :, -1] = torch.arange(loader.frames)[:, None]  # Adding time
    pos_array[0, :, :2] = torch.tensor(init, dtype=torch.float32)
    return pos_array


def _localize(
    loader: DataLoader,
    preprocessor: Callable,
    localizer: Callable,
    locations: torch.Tensor,
    n_frames: int,
    device: str,
    max_change: float,
):

    initializing_frame = None
    for _, (frame_idx, image) in takewhile(
        lambda x: x[0] <= n_frames, enumerate(loader)
    ):
        image = image.to(device, non_blocking=True)
        if initializing_frame is None:
            initializing_frame = np.maximum(frame_idx - 1, 0)
        new_positions, delta_position = localizer(
            preprocessor(image), locations[initializing_frame, :, :2]
        )

        if delta_position < max_change:
            locations[frame_idx, :, :2] = new_positions
            initializing_frame = frame_idx
        if frame_idx % 1000 == 0:
            print(f"Done with frame {frame_idx}")
    loader.stop()
    return locations


def _multi_run(
    files: Iterable[DataLoader],
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

    positions_list = []
    positions = None
    for file in files:
        loader = DataLoader(file)
        if positions is None:
            positions = _initialize(
                loader, initial_preprocessor, initial_localizer, n_ini, device
            )
        else:
            positions = _multi_initialize(loader, positions[-1, :, :2], device)
        # Now run main localizer

        positions = _localize(
            loader,
            main_preprocessor,
            main_localizer,
            positions,
            n_frames,
            device,
            max_change,
        )
        loader.stop()
        positions_list.append(torch.clone(positions))

    # Turn into dataframe
    positions = torch.cat(positions_list, axis=0)
    positions[:, :, -1] = torch.arange(positions.shape[0])[:, None]  # adding imte
    non_zero_frames = torch.sum(positions[:, :, :2], axis=[1, 2]) != 0
    positions = tracker(positions[non_zero_frames])
    df = post_process(positions, n_arenas)
    return df
