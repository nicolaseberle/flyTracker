from typing import Callable, Tuple
import numpy as np
import cv2 as cv
import pandas as pd
from itertools import count

from .preprocessing import construct_undistort_map, preprocessing
from .postprocessing import post_process
from .tracking import (
    blob_detector_localization,
    localize_kmeans,
    localize_kmeans_jax,
    localize_kmeans_torch,
    hungarian,
)


def run(
    movie_path: str,
    mask: np.ndarray,
    n_arenas: int,
    mapping_folder: str,
    n_frames: int = None,
    n_ini: int = 100,
    method="kmeans",
) -> pd.DataFrame:
    """Runs the whole pipeline. I.e preprocessing, localizing and postprocessing."""
    # Constructing loader
    capture = cv.VideoCapture(movie_path)

    image_size = (
        int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)),
    )
    mapping = construct_undistort_map(image_size, mapping_folder)
    loader = lambda: preprocessing(capture.read()[1], mapping=mapping, mask=mask)

    if method == "kmeans":
        localization_fn = localize_kmeans
    elif method == "kmeans_jax":
        localization_fn = localize_kmeans_jax
    elif method == "kmeans_pytorch":
        localization_fn = localize_kmeans_torch
    elif method == "GMM":
        raise NotImplementedError
    else:
        raise NotImplementedError

    # Actual logic
    n_flies, initial_locations, initial_frame = initialize(loader, n_ini)
    locs = localize(loader, localization_fn, initial_locations, n_frames=n_frames)
    df = post_process(locs, initial_frame, n_arenas=n_arenas)
    return df


def initialize(loader: Callable, n_frames: int) -> Tuple[int, np.ndarray, int]:
    """Find flies using blob detector and
    calulate number of flies."""
    n_blobs = []
    for frame_idx in count():
        image = loader()
        locations = blob_detector_localization(image)
        n_blobs.append(locations.shape[0])

        if len(n_blobs) >= n_frames:
            n_flies = int(np.median(n_blobs))
            if n_blobs[-1] == n_flies:
                break
    # pluse on cause the next one is the first
    initial_frame = frame_idx + 1
    return n_flies, locations, initial_frame


def localize(
    loader: Callable,
    localization_fn: Callable,
    initial_position: np.ndarray,
    n_frames: int = None,
) -> np.ndarray:
    "Track flies using kmeans"
    data = [initial_position]
    for idx in count():
        try:
            image = loader()
        except:
            break  # finished

        data.append(hungarian(localization_fn(image, data[-1]), data[-1]))

        if idx % 1000 == 0:
            print(f"Done with frame {idx}")
        if idx + 1 == n_frames:
            break  # max number of frames

    return data[1:]
