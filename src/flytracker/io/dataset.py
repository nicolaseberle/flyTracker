import torch
import cv2
from torchvision import io
from torchvision.transforms.functional import rgb_to_grayscale
from ..preprocessing.preprocessing import preprocessing


class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, movie_path, mapping_folder, mask):
        super().__init__()
        self.reader = cv2.VideoCapture(movie_path)
        self.preprocessor = preprocessing(mask, mapping_folder)

    def __iter__(self):
        return self

    def __next__(self):
        succes, image = self.reader.read()
        if succes is False:
            raise StopIteration

        return self.preprocessor(image)


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
