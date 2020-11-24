import torch
import cv2 as cv
import numpy as np


class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, path: str, mask, threshold=120) -> None:
        super().__init__()
        self.video_path = path
        self.capture = cv.VideoCapture(self.video_path)
        self.threshold = threshold
        self.mask = mask.astype("uint8")

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        # Loading image
        succes, image = self.capture.read()
        if succes is False:
            raise StopIteration
        fly_pixels = self.preprocess(image)

        return fly_pixels

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        # Turn it into grayscale
        processed_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Thresholding
        processed_image = cv.threshold(
            processed_image, self.threshold, 255, cv.THRESH_BINARY_INV
        )[1]
        # Applying mask
        processed_image = processed_image * self.mask
        # Getting locations of pixels
        fly_pixels = cv.findNonZero(processed_image).squeeze()
        fly_pixels = torch.tensor(fly_pixels)
        return fly_pixels


if __name__ == "__main__":
    from flytracker.utils import FourArenasQRCodeMask
    from torch.utils.data import DataLoader

    mask = FourArenasQRCodeMask().mask
    path = "/home/gert-jan/Documents/flyTracker/data/movies/4arenas_QR.h264"

    dataset = VideoDataset(path, mask)
    loader = DataLoader(dataset, batch_size=1, pin_memory=True)

    for batch_idx, sample in enumerate(loader):
        sample = sample.cuda(non_blocking=True)
        if batch_idx == 1000:
            break
