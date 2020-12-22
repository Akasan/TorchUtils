import torch
import torchvision
import warnings
from typing import Tuple
import numpy as np
warnings.simplefilter("ignore")


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor, transform: torchvision.transforms.transforms = None):
        self.transform = transform
        self.data = data.detach().numpy()
        self.labels = labels.detach().numpy()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx) -> Tuple[np.ndarray, int]:
        data = self.data[idx]
        label = self.labels[idx]

        if self.transform is not None:
            data = self.transform(data)

        return data, label

    def size(self) -> tuple:
        return self.data.size()