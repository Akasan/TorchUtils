from PIL import Image
from glob import glob
import torch
import numpy as np
import os
import random
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from ._LoaderGenerator import generate_dataloader as gd
from ._SplitDataset import split_dataset


# TODO 複数ラベルバージョン(CSVで管理)も作る
class SingleFolderSingleLabelDataset(torch.utils.data.Dataset):
    """You can make dataset for one specified folder

    Examples:
    ---------
        - folder tree
            A--1.jpg
             |-2.jpg ...
            B--1.jpg
             |-2.jpg ...

        >>> datasetA = SingleFolderSingleLabelDataset("A")
        >>> datasetB = SingleFolderSingleLabelDataset("B")
    """

    def __init__(self, dir_name, transform=None, label=None, ext="jpg", shuffle=True):
        """
        Arguments:
        ----------
            dir_name {str} -- root folder of containing images.
                              Please set this as a absolute path.

        Keyword Arguments:
        ------------------
            transform {torchvision.transforms} -- transforms (default: None)
                                                  If transforms is None, the following transform will be used.
                                                  transforms.Compose([transforms.ToTensor()])
            label {int} -- if you dont specified this argument, the dir_name will be set as label (default: None)
            ext {str} -- image file's extension (default: "jpg")
            shuffle {bool} -- set True when you want to shuffle the order of images (default: True)

        Examples:
        ---------
            >>> custom_dataset = SingleFolderSingleLabelDataset(dir_name="./data",
        """
        self.DIR_NAME = dir_name

        if label is None:
            self.LABEL = dir_name.replace("/", "\\").split("\\")[-1]
        else:
            self.LABEL = label

        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])

        self.transform = transform
        files = os.listdir(os.path.abspath(dir_name))
        self.files = [
            os.path.join(os.path.abspath(dir_name), f)
            for f in files
            if f.split(".")[-1] == ext
        ]
        self.length = len(self.files)

        if shuffle:
            random.shuffle(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        img = Image.open(self.files[i])
        img = self.transform(img)
        return img, self.LABEL


def generate_dataset(folder_path, transform):
    """generate dataset from folder

    Arguments:
    ----------
        folder_path {str} -- folder path
        transform {torchvision.transforms} -- transform

    Returns:
    --------
        {torch.utils.data.Dataset} -- dataset

    Examples:
    ---------
        >>> transform = transforms.Compose([transforms.ToTensor()])
        >>> dataset = generate_dataset("root", transform)
    """
    return ImageFolder(folder_path, transform)


def generate_dataloader(
    folder_path,
    transform=None,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    is_single_label=False,
    ext="jpg",
    label=0,
):
    """generate dataloader from folder

    Arguments:
    ----------
        folder_path {str} -- folder path

    Keyword Arguments:
    ------------------
        transform {torchvision.transforms} -- transform (default: None)
        batch_size {int} -- batch size (default: 128)
        shuffle {bool} -- whether data will be shuffled or not (default: True)
        num_workers {int} -- the number of workers (default: 2)
        is_single_label {bool} -- set as True when specified folder only contains data which has the same label (default: False)

    Returns:
    --------
        {torch.utils.data.DataLoader} -- dataloader

    Examples:
    ---------
        >>> dataloader = generate_dataloader("root")
    """
    if is_single_label:
        dataset = SingleFolderSingleLabelDataset(
            folder_path, transform, label, ext, shuffle
        )
    else:
        dataset = generate_dataset(folder_path, transform)

    return gd(dataset, batch_size, shuffle, num_workers)


def generate_dataloader_with_val(
    folder_path,
    transform=None,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    validation_rate=0.2,
):
    """generate dataloder from folder with validation

    Arguments:
    ----------
        folder_path {str} -- folder path

    Keyword Arguments:
    ------------------
        transform {torchvision.transforms} -- transform (default: None)
        batch_size {int} -- batch size (default: 128)
        shuffle {bool} -- whether data will be shuffled or not (default: True)
        num_workers {int} -- the number of workers (default: 2)
        validation_rate {float} -- the rate of validation (default: 0.2)

    Returns:
    --------
        {torch.utils.data.DataLoader} -- dataloader for training
        {torch.utils.data.DataLoader} -- dataloader for validation
    """
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
    pass
