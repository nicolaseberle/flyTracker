import numpy as np
import cv2 as cv
import pandas as pd

from sklearn.cluster import k_means


def post_process(
    locations: np.ndarray, initial_frame: int, n_arenas: int
) -> pd.DataFrame:
    """ Post processing: turns into dataframe, sorts per arena."""
    n_frames = len(locations)
    n_flies = len(locations[0])
    identities = (np.arange(n_flies)[None, :] * np.ones((n_frames, n_flies))).reshape(
        -1, 1
    )  # we get free tracking from the kmeans
    frames = (
        np.arange(initial_frame, n_frames + initial_frame)[:, None]
        * np.ones((n_frames, n_flies))
    ).reshape(-1, 1)
    df = pd.DataFrame(
        np.concatenate([frames, identities, np.concatenate(locations, axis=0)], axis=1),
        columns=["frame", "ID", "x", "y"],
    )

    # Localizing flies per arena
    df["arena"] = find_arena(df, n_arenas)
    return df


def find_arena(df, n_arenas):
    x_ave = df.pivot_table(index="ID", columns="frame", values="x").mean(axis=1)
    y_ave = df.pivot_table(index="ID", columns="frame", values="y").mean(axis=1)
    labels = k_means(np.stack([x_ave, y_ave], axis=1), n_arenas)[1]
    arena = (labels[None, :] * np.ones((df.frame.unique().size, 1))).flatten()
    return arena
