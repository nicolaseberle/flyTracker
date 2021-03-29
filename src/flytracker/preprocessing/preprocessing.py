import numpy as np
import torch
from torchvision.transforms.functional import rgb_to_grayscale
import cv2 as cv
from .undistort_mapping import construct_undistort_map
from functools import partial


def preprocessing(mask, mapping_folder):
    def _preprocessing(mask, mapping, image) -> np.ndarray:
        """Preprocesses image to make it ready for kmeans."""
        # TODO: Turn mapping into generic function.
        processed_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        processed_image = cv.remap(processed_image, *mapping, cv.INTER_LINEAR)
        processed_image[~mask] = 255
        return processed_image

    mapping = construct_undistort_map(mask.shape[::-1], mapping_folder)
    return partial(_preprocessing, mask, mapping)


def preprocessing_torch(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Preprocesses image to make it ready for kmeans."""
    image = torch.tensor(image)
    image = rgb_to_grayscale(image)
    image = torch.where(mask, image.squeeze(), torch.tensor(255, dtype=torch.uint8))
    return image

