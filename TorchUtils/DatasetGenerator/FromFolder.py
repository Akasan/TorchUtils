import torch
import numpy as np
import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from ._LoaderGenerator import generate_dataloader as gd
from ._SplitDataset import split_dataset


def generate_dataset(folder_path, transform):
    return ImageFolder(folder_path, transform)


def generate_dataloader(folder_path, transform=None, batch_size=128, shuffle=True, num_workers=2):
    dataset = generate_dataset(folder_path, transform)
    loader = gd(dataset, batch_size, shuffle, num_workers)
    return loader


def generate_dataloader_with_val(folder_path, transform=None, batch_size=128, shuffle=True, num_workers=2, validation_rate=0.2):
    dataset = generate_dataset(folder_path, transform)
    n_samples = len(dataset)
    val_size = int(n_samples * validation_rate)
    train_size = n_samples - val_size
    train_dataset, val_dataset = split_dataset(dataset, validation_rate)
    train_loader = gd(train_dataset, batch_size, shuffle, num_workers)
    val_loader = gd(val_dataset, batch_size, shuffle, num_workers)
    return train_loader, val_loader


def to_cv2_format(img):
    pass


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    loader = generate_dataset_loader("C:\\Users\\chino\\Downloads\\31797_40972_bundle_archive\\train", transform, batch_size=1)