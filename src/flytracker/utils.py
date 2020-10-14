import numpy as np


class FourArenasQRCodeMask:
    """Contains mask for four arenas with QR code."""
    @property
    def mask(self):
        # Building mask, this will be better later
        mask = np.ones((1080, 1280), dtype=np.bool) # assumes 1080 x 1280 resolution
        mask[:180, :300] = 0 # upper left corner
        mask[-230:, :300] = 0 # lower left corner
        mask[-230:, -300:] = 0 # lower right corner
        mask[:180, -300:] = 0 # lower right corner
        mask[:70, :] = 0
        mask[-120:, :] = 0
        mask[:, :180] = 0
        mask[:, -200:] = 0
        
        return mask
