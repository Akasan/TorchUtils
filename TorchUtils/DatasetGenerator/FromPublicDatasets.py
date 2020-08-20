import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


__DATASET = {
    "MNIST": torchvision.datasets.MNIST,
    "CIFAR10": torchvision.datasets.CIFAR10,
    "CIFAR100": torchvision.datasets.CIFAR100,
    "CocoCaptions": torchvision.datasets.CocoCaptions,
    "CocoDetection": torchvision.datasets.CocoDetection,
    "EMNIST": torchvision.datasets.EMNIST,
    "FashinMNIST": torchvision.datasets.FashionMNIST
}

def _load(datasets, root="./data", download=True, transform=None, batch_size=128, shuffle=True, num_workers=2):
    train_dataset = None
    test_dataset = None

    if transform is not None:
        train_dataset = datasets(root=root,
                                 train=True,
                                 download=download,
                                 transform=transform)

        test_dataset = datasets(root=root,
                                train=False,
                                download=download,
                                transform=transform)

    else:
        train_dataset = datasets(root=root,
                                 train=True,
                                 download=download)

        test_dataset = datasets(root=root,
                                train=False,
                                download=download)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)

    return train_loader, test_loader


def load_public_datasets(name="MNIST", root="./data", download=True, transform=None, batch_size=128, shuffle=True, num_workers=2):
    datasets = __DATASET[name]
    return _load(datasets, root="./data", download=True, transform=None, batch_size=128, shuffle=True, num_workers=2)


def get_public_datasets_list():
    print(list(__DATASET.keys()))


if __name__ == "__main__":
    train_loader, test_loader = load_public_datasets("CocoDetection")