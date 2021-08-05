import torch
import cv2
from torchvision import io
from .videoreader import VideoReader


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, path, **reader_kwargs):
        super().__init__(
            VideoDataset(path, **reader_kwargs), batch_size=None, pin_memory=True,
        )

    def stop(self):
        self.dataset.reader.release()

    @property
    def frames(self,):
        return int(self.dataset.reader.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def current_frame(self,):
        return int(self.dataset.reader.get(cv2.CAP_PROP_POS_FRAMES))

    @property
    def remaining_frames(self,):
        return int(self.frames - self.current_frame)


class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, **reader_kwargs):
        super().__init__()
        self.reader = cv2.VideoCapture(path, **reader_kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        frame_idx = int(self.reader.get(cv2.CAP_PROP_POS_FRAMES))
        succes, image = self.reader.read()
        if succes is False:
            self.reader.release()
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
