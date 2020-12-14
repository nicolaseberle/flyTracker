import torch
import cv2 as cv
import numpy as np
from os.path import join
from typing import Tuple


class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, path: str, mask, threshold=120) -> None:
        super().__init__()
        self.video_path = path
        self.capture = cv.VideoCapture(self.video_path)
        self.threshold = threshold
        self.mask = mask.astype("uint8")

        self.image_size = (
            int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT)),
        )
        self.undistort_map = self.construct_undistort_map(self.image_size)

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        # Loading image
        succes, image = self.capture.read()
        if succes is False:
            raise StopIteration
        fly_pixels = self.preprocess(
            image, self.mask, self.threshold, self.undistort_map
        )

        return fly_pixels

    @staticmethod
    def preprocess(
        image: np.ndarray,
        mask: np.ndarray,
        threshold: int,
        undistort_map: Tuple[np.ndarray, np.ndarray],
    ) -> torch.Tensor:

        # Turn it into grayscale
        processed_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Thresholding
        processed_image = cv.threshold(
            processed_image, threshold, 255, cv.THRESH_BINARY_INV
        )[1]

        # Applying mask
        processed_image = processed_image * mask

        # Undistorting the image
        processed_image = cv.remap(processed_image, *undistort_map, cv.INTER_LINEAR)

        # Finding locations of non-zero pixels
        fly_pixels = cv.findNonZero(processed_image).squeeze()

        fly_pixels = torch.tensor(fly_pixels)
        return fly_pixels

    @staticmethod
    def construct_undistort_map(
        image_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:

        folder = "/home/gert-jan/Documents/flyTracker/data/distortion_maps/"
        mtx = np.load(join(folder, "mtx_file.npy"))
        dist = np.load(join(folder, "dist_file.npy"))
        newcameramtx = np.load(join(folder, "newcameramtx_file.npy"))

        map = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, image_size, 5)
        return map


if __name__ == "__main__":
    from flytracker.utils import FourArenasQRCodeMask
    from torch.utils.data import DataLoader
    import time

    mask = FourArenasQRCodeMask().mask
    path = "/home/gert-jan/Documents/flyTracker/data/movies/4arenas_QR.h264"

    dataset = VideoDataset(path, mask)
    loader = DataLoader(dataset, batch_size=1, pin_memory=True)

    t_start = time.time()
    for batch_idx, sample in enumerate(loader):
        # sample = sample.cuda(non_blocking=True)
        if batch_idx == 1000:
            break
    t_end = time.time()
    print(t_end - t_start)
