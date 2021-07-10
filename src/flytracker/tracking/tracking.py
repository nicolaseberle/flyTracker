import numpy as np
from .hungarian import hungarian
from itertools import accumulate
import torch


def tracking(locations, ordering_fn=hungarian):
    if isinstance(locations, torch.Tensor):
        locations = locations.cpu().numpy()

    ordered_locs = list(accumulate(locations, ordering_fn))
    return ordered_locs
