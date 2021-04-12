import torch
import cv2
from torchvision import io
from .videoreader import VideoReader


class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, parallel=False):
        super().__init__()
        self.parallel = parallel
        if self.parallel:
            self.reader = VideoReader(path)
        else:
            self.reader = cv2.VideoCapture(path)

    def __iter__(self):
        return self

    def __next__(self):
        succes, image = self.reader.read()
        if succes is False:
            raise StopIteration

        return image

    def set_frame(self, frame_idx):
        """Set iterator to certain frame so next load
        is frame_idx."""
        self.reader.set(1, frame_idx)


class TorchVideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, path):
        super().__init__()
        self.reader = io.VideoReader(path)

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        # opencv loads differently and we permute for that
        image = next(self.reader)["data"].permute(1, 2, 0)
        if image is None:
            raise StopIteration
        return image
