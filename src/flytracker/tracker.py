import numpy as np
import pandas as pd
from typing import Callable, Tuple
from itertools import count
from .io.loader import opencv_loader
from .localization.blob.blob import blob_detector_localization
from .localization.kmeans.kmeans import localize_kmeans
from .analysis.postprocessing import post_process
from .tracking.tracking import tracking


def run(
    movie_path: str,
    mask: np.ndarray,
    n_arenas: int,
    mapping_folder: str,
    n_frames: int = None,
    n_ini: int = 100,
) -> pd.DataFrame:
    """User facing run function with sensible standard settings."""

    loader = opencv_loader(movie_path, mapping_folder, mask)
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
    loader: Callable,
    initial_localizer: Callable,
    main_localizer: Callable,
    tracker: Callable,
    post_process: Callable,
    n_arenas: int,
    n_frames=None,
    n_ini=100,
):
    initial_position, initial_frame = _initialize(loader, initial_localizer, n_ini)
    locations = _localize(loader, main_localizer, initial_position, n_frames)
    ordered_locations = tracker(locations)
    df = post_process(ordered_locations, initial_frame, n_arenas)
    return df


def _initialize(
    loader: Callable, localizer: Callable, n_frames: int
) -> Tuple[np.ndarray, int]:
    n_blobs = []
    for frame_idx in count():
        locations = localizer(loader())
        n_blobs.append(locations.shape[0])

        if len(n_blobs) >= n_frames:
            n_flies = int(np.median(n_blobs))
            if n_blobs[-1] == n_flies:
                break

    return locations, frame_idx


def _localize(
    loader: Callable,
    localizer: Callable,
    initial_position: np.ndarray,
    n_frames: int = None,
) -> np.ndarray:

    locations = [initial_position]
    for idx in count():
        image = loader()
        if image is None:
            break

        locations.append(localizer(image, locations[-1]))

        if idx % 1000 == 0:
            print(f"Done with frame {idx}")
        if idx + 1 == n_frames:
            break  # max number of frames
    return locations
