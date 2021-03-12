#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from flytracker.utils import FourArenasQRCodeMask
from torch.utils.data import DataLoader
from itertools import takewhile
import matplotlib.pyplot as plt
from torchvision.transforms.functional import rgb_to_grayscale, to_tensor
import cv2 as cv


# In[22]:


class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, device="cuda"):
        super().__init__()
        self.capture = cv.VideoCapture(path)
        self.device = device

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        # Loading image
        succes, image = self.capture.read()
        if succes is False:
            raise StopIteration

        image = torch.tensor(image)
        image = torch.movedim(torch.tensor(image), -1, 0)
        image = rgb_to_grayscale(image).squeeze()
        return image


mask = FourArenasQRCodeMask().mask
path = "/home/gert-jan/Documents/flyTracker/data/movies/4arenas_QR.h264"

dataset = VideoDataset(path, device="cuda")
loader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=4)


for batch_idx, batch in enumerate(loader):
    image = batch.cuda(non_blocking=True)
    if batch_idx % 100 == 0:
        print(f"Loaded {batch_idx}, {image.device}")
    if batch_idx == 1000:
        break

