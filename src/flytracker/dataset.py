import torch
import cv2 as cv
import numpy as np
from flytracker.preprocessing import preprocessing


class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, mask):
        super().__init__()
        self.capture = cv.VideoCapture(path)
        self.mask = torch.tensor(mask, dtype=torch.bool)

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        # Loading image
        # TODO: Use pytorch vision loader instead of opencv
        succes, image = self.capture.read()
        if succes is False:
            raise StopIteration
        # because we use opencv
        image = np.moveaxis(image, -1, 0)
        processed_image = preprocessing(image, self.mask)
        return processed_image

