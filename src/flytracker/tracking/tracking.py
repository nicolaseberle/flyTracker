import numpy as np
from .hungarian import hungarian
from itertools import accumulate


def tracking(locations, ordering_fn=hungarian):
    locs = np.stack(locations, axis=0)
    ordered_locs = list(accumulate(locs, ordering_fn))
    return ordered_locs
