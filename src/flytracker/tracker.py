from typing import Callable, Tuple
import numpy as np
import cv2 as cv
import pandas as pd
from itertools import count

from .preprocessing import construct_undistort_map, preprocessing
from .postprocessing import post_process
from .tracking import blob_detector_localization, localize_kmeans, hungarian


def run(
    movie_path: str,
    mask: np.ndarray,
    n_arenas: int,
    mapping_folder: str,
    n_frames: int = None,
    n_ini: int = 100,
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

    # Actual logic
    n_flies, initial_locations, initial_frame = initialize(loader, n_ini)
    locs = localize(loader, initial_locations, n_frames=n_frames)
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

    return n_flies, locations, frame_idx


def localize(
    loader: Callable, initial_position: np.ndarray, n_frames: int = None
) -> np.ndarray:
    "Track flies using kmeans"
    locations = [initial_position]
    for idx in count():
        try:
            image = loader()
        except:
            break  # finished

        locations.append(
            hungarian(localize_kmeans(image, locations[-1]), locations[-1])
        )

        if idx % 1000 == 0:
            print(f"Done with frame {idx}")
        if idx + 1 == n_frames:
            break  # max number of frames
    return locations
