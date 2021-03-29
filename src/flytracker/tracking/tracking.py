import numpy as np
from .hungarian import hungarian
from itertools import accumulate
import torch


def tracking(locations, ordering_fn=hungarian):
    if isinstance(locations[-1], torch.Tensor):
        locs = torch.stack(locations, dim=0).cpu().numpy()
    else:
        locs = np.stack(locations, axis=0)

    ordered_locs = list(accumulate(locs, ordering_fn))
    return ordered_locs
