from ..io.dataset import DataLoader
import torch
from torchvision.transforms.functional import rgb_to_grayscale
from ..localization.blob.blob import localize_blob


def load_frame(path: str, frame: int, color=False):
    """Load frame of video and turn into grayscale."""
    loader = DataLoader(path, parallel=False)
    loader.dataset.set_frame(frame)

    _, image = next(enumerate(loader))
    if not color:
        image = rgb_to_grayscale(image.permute(2, 0, 1)).squeeze()

    return image


def test_mask(image: torch.Tensor, mask: torch.Tensor):
    """Applies the mask to a video - can be used to easily
    test and plot."""
    masked_image = torch.where(mask, image, torch.tensor(255, dtype=torch.uint8))
    return masked_image


def test_threshold(image: torch.Tensor, threshold: int):
    """Threshold given image."""
    return image < threshold


def test_blob_detector(image: torch.Tensor, blob_detector_params):
    """Applies blob detector with given params to given image."""
    return localize_blob(blob_detector_params)(image.numpy())
