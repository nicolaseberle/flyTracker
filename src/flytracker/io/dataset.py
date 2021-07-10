import torch
import cv2
from torchvision import io
from .videoreader import VideoReader


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, path, parallel, **reader_kwargs):
        self.parallel = parallel
        super().__init__(
            VideoDataset(path, self.parallel, **reader_kwargs),
            batch_size=None,
            pin_memory=True,
        )

    def stop(self):
        if self.parallel:
            self.dataset.reader.stop()


class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, parallel=False, **reader_kwargs):
        super().__init__()
        self.parallel = parallel
        if self.parallel:
            self.reader = VideoReader(path, **reader_kwargs)
        else:
            self.reader = cv2.VideoCapture(path, **reader_kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.parallel:
            frame_idx = self.reader.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            frame_idx = 0
        succes, image = self.reader.read()
        if succes is False:
            raise StopIteration

        return frame_idx, image

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
