import numpy as np
import pandas as pd
from sklearn.cluster import k_means
import torch
from itertools import accumulate
from flytracker.tracking import hungarian


def post_process(locations, initial_frame, n_arenas):
    # Getting some useful properties
    n_frames = len(locations)
    n_flies = len(locations[0])

    # finding identities and frames
    identities = np.tile(np.arange(n_flies), n_frames)[:, None]
    frames = np.repeat(np.arange(initial_frame, n_frames + initial_frame), n_flies)[
        :, None
    ]

    # Making dataframe
    df = pd.DataFrame(
        np.concatenate([frames, identities], axis=1), columns=["frame", "ID"]
    )
    # Actually performing the tracking
    locs = torch.stack(locations, axis=0).cpu().numpy()
    ordered_locs = list(accumulate(locs, func=lambda x, y: hungarian(y, x)))
    df[["x", "y"]] = np.concatenate(ordered_locs, axis=0)

    # Adding arenas and reordering flies
    df = find_arena(df, n_arenas)
    df = order_flies(df)

    return df


def find_arena(df, n_arenas):
    n_frames = df.frame.unique().size

    # Finding arenas and labels
    x_ave = df.pivot_table(index="ID", columns="frame", values="x").mean(axis=1)
    y_ave = df.pivot_table(index="ID", columns="frame", values="y").mean(axis=1)
    arena_centers, labels, _ = k_means(np.stack([x_ave, y_ave], axis=1), n_arenas)

    # Ordering arena centres left to right and modifying labels
    arena_centers = np.around(arena_centers, decimals=-2)  # roughly same quarter
    ordering = np.lexsort((arena_centers[:, 0], arena_centers[:, 1]))
    ordered_labels = [ordering[idx] for idx in labels]
    df["arena"] = np.tile(ordered_labels, n_frames)

    return df


def order_flies(df):
    # New IDs so flies [0, n_flies] are in arena 0
    # [n_flies, 2 x n_flies] in arena 1 etc.
    n_flies = df.ID.unique().size
    n_frames = df.frame.unique().size

    df = df.sort_values(by=["frame", "arena", "ID"])
    df["ID"] = np.tile(np.arange(n_flies), n_frames)
    df = df.sort_values(by=["frame", "ID"], ignore_index=True)

    return df
