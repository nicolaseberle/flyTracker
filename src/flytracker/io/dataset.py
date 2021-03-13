import torch
import numpy as np
import cv2
from .videoreader import VideoReader
from torchvision import io
from torchvision.transforms.functional import rgb_to_grayscale, to_pil_image, to_tensor


class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, mask, parallel=False, *parallel_kwargs):
        super().__init__()
        if parallel:
            self.reader = VideoReader(path, *parallel_kwargs)
        else:
            self.reader = cv2.VideoCapture(path)

        self.mask = torch.tensor(mask, dtype=torch.bool)

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        succes, image = self.reader.read()
        if succes is False:
            raise StopIteration
        image = torch.tensor(image)
        image = rgb_to_grayscale(image.permute(2, 0, 1))
        image = torch.where(self.mask, image, torch.tensor(255, dtype=torch.uint8))
        return image.squeeze()


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
