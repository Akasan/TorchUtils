import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from ._LoaderGenerator import generate_dataloader
from ._SplitDataset import split_dataset


__DATASET = {
    "MNIST": torchvision.datasets.MNIST,
    "CIFAR10": torchvision.datasets.CIFAR10,
    "CIFAR100": torchvision.datasets.CIFAR100,
    "CocoCaptions": torchvision.datasets.CocoCaptions,
    "CocoDetection": torchvision.datasets.CocoDetection,
    "EMNIST": torchvision.datasets.EMNIST,
    "FashionMNIST": torchvision.datasets.FashionMNIST
}


def _load(datasets, root="./data", download=True, transform=None, batch_size=128, shuffle=True, num_workers=2):
    train_dataset = None
    test_dataset = None

    train_dataset = datasets(root=root,
                             train=True,
                             download=download,
                             transform=transform)

    test_dataset = datasets(root=root,
                            train=False,
                            download=download,
                            transform=transform)

    train_loader = generate_dataloader(train_dataset, batch_size, shuffle, num_workers)
    test_loader = generate_dataloader(test_dataset, batch_size, shuffle, num_workers)

    return train_loader, test_loader


def _load_with_val(datasets, root="./data", download=True, transform=None, batch_size=128, shuffle=True, num_workers=2, validation_rate=0.2):
    train_dataset = None
    test_dataset = None

    train_dataset_tmp = datasets(root=root,
                                 train=True,
                                 download=download,
                                 transform=transform)

    train_dataset, val_dataset = split_dataset(train_dataset_tmp, validation_rate)

    test_dataset = datasets(root=root,
                            train=False,
                            download=download,
                            transform=transform)

    train_loader = generate_dataloader(train_dataset, batch_size, shuffle, num_workers)
    val_loader = generate_dataloader(val_dataset, batch_size, shuffle, num_workers)
    test_loader = generate_dataloader(test_dataset, batch_size, shuffle, num_workers)

    return train_loader, val_loader, test_loader



def load_public_dataset(name="MNIST", root="./data", download=True,
                        transform=None, batch_size=128, shuffle=True, num_workers=2):
    """ load_public_datasets

    Keyword Arguments:
    ------------------
        name {str} -- public dataset's name (default: "MNIST")
        root {str} -- root directory where dataset is (default: "./data")
        download {bool} -- True when you want to download dataset (default: True)
        transform {[type]} -- transform (default: None)
        batch_size {int} -- batch size (default: 128)
        shuffle {bool} -- True when you want to shuffle dataset (default: True)
        num_workers {int} -- num workers (default: 2)

    Returns:
    --------
        {torch.util.data.dataloader.DataLoader} -- train dataset's loader
        {torch.util.data.dataloader.DataLoader} -- test dataset's loader

    Examples:
    ---------
        >>> train_loader_mnist, test_loader_mnist = load_public_datasets()
        >>> train_loader_mnist, test_loader_mnist = load_public_datasets("MNIST")
        >>> train_loader_cifar10, test_loader_cifer10 = load_public_datasets("CIFAR10")
    """
    dataset = __DATASET[name]
    return _load(dataset, root="./data", download=download, transform=transform,
                 batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def load_public_dataset_with_val(name="MNIST", root="./data", download=True,
                                 transform=None, batch_size=128, shuffle=True,
                                 num_workers=2, validation_rate=0.2):
    """ load_public_datasets

    Keyword Arguments:
    ------------------
        name {str} -- public dataset's name (default: "MNIST")
        root {str} -- root directory where dataset is (default: "./data")
        download {bool} -- True when you want to download dataset (default: True)
        transform {[type]} -- transform (default: None)
        batch_size {int} -- batch size (default: 128)
        shuffle {bool} -- True when you want to shuffle dataset (default: True)
        num_workers {int} -- num workers (default: 2)

    Returns:
    --------
        {torch.util.data.dataloader.DataLoader} -- train dataset's loader
        {torch.util.data.dataloader.DataLoader} -- test dataset's loader

    Examples:
    ---------
        >>> train_loader_mnist, test_loader_mnist = load_public_datasets()
        >>> train_loader_mnist, test_loader_mnist = load_public_datasets("MNIST")
        >>> train_loader_cifar10, test_loader_cifer10 = load_public_datasets("CIFAR10")
    """
    dataset = __DATASET[name]
    return _load_with_val(dataset, root="./data", download=download, transform=transform,
                          batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, validation_rate=validation_rate)


def get_public_datasets_list():
    """ get_public_datasets_list

    Examples:
    ---------
        >>> get_public_datasets_list()
        ... ['MNIST', 'CIFAR10', 'CIFAR100', 'CocoCaptions', 'CocoDetection', 'EMNIST', 'FashionMNIST']
    """
    return list(__DATASET.keys())


if __name__ == "__main__":
    train_loader, test_loader = load_public_dataset_with_val("MNIST")