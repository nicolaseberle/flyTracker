import torch
import numpy as np
import cv2 as cv
from flytracker.preprocessing import preprocessing
from flytracker.videoreader import VideoReader
from torchvision import io
from torchvision.transforms.functional import rgb_to_grayscale


class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, mask, parallel=False, *parallel_kwargs):
        super().__init__()
        if parallel:
            self.reader = VideoReader(path, *parallel_kwargs)
        else:
            self.reader = cv.VideoCapture(path)
        self.mask = torch.tensor(mask, dtype=torch.bool)

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        succes, image = self.reader.read()
        if succes is False:
            raise StopIteration
        # because we use opencv
        image = np.moveaxis(image, -1, 0)
        image = torch.tensor(image)
        image = rgb_to_grayscale(image)
        processed_image = torch.where(
            self.mask, image.squeeze(), torch.tensor(255, dtype=torch.uint8)
        )
        return processed_image


class TorchVideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, mask):
        super().__init__()
        self.reader = io.VideoReader(path)
        self.mask = torch.tensor(mask, dtype=torch.bool)

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        # Loading image
        image = next(self.reader)["data"]
        image = rgb_to_grayscale(image).squeeze()
        image = torch.where(self.mask, image, torch.tensor(255, dtype=torch.uint8))
        return image
