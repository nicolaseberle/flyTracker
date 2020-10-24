import numpy as np
import pandas as pd

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

@pd.api.extensions.register_dataframe_accessor("groups")
class Groups:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        
    @staticmethod
    def _validate(obj):
        pass

    @property
    def coordinates(self):
        # return the geographic center point of this DataFrame
        return self._obj[['x', 'y']]
    
    @property
    def velocities(self):
        return self.obj[['v_x', 'v_y']]