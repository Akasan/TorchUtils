import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from ._LoaderGenerator import generate_dataloader
from ._CustomDataset import CustomDataset
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


__DEFAULT_TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])


def _generate_dataset(datasets, root, download, transform, train):
    return datasets(root=root, train=train, download=download, transform=transform)


def _generate_loader(datasets, root="./data", download=True, transform=None, batch_size=128, shuffle=True, num_workers=2):
    train_dataset = None
    test_dataset = None
    train_dataset = _generate_dataset(datasets, root, download, transform, True)
    test_dataset = _generate_dataset(datasets, root, download, transform, False)
    train_loader = generate_dataloader(train_dataset, batch_size, shuffle, num_workers)
    test_loader = generate_dataloader(test_dataset, batch_size, shuffle, num_workers)
    return train_loader, test_loader


def _generate_loader_with_val(datasets, root="./data", download=True, transform=None, batch_size=128, shuffle=True, num_workers=2, validation_rate=0.2):
    train_dataset = None
    test_dataset = None
    train_dataset_tmp = _generate_dataset(datasets, root, download, transform, True)
    test_dataset = _generate_dataset(datasets, root, download, transform, False)
    train_dataset, val_dataset = split_dataset(train_dataset_tmp, validation_rate)
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
    return _generate_loader(dataset, root="./data", download=download, transform=transform,
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
        validation_rate {float} -- validation rate (default: 0.2)

    Returns:
    --------
        {torch.util.data.dataloader.DataLoader} -- train dataset's loader
        {torch.util.data.dataloader.DataLoader} -- validation dataset's loader
        {torch.util.data.dataloader.DataLoader} -- test dataset's loader

    Examples:
    ---------
        >>> train_loader_mnist, val_loader, test_loader_mnist = load_public_datasets()
        >>> train_loader_mnist, val_loader, test_loader_mnist = load_public_datasets("MNIST")
        >>> train_loader_cifar10, val_loader_cifar10, test_loader_cifer10 = load_public_datasets("CIFAR10")
    """
    dataset = __DATASET[name]
    return _generate_loader_with_val(dataset, root="./data", download=download, transform=transform,
                                     batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, validation_rate=validation_rate)


def _append_data(whole_data, whole_labels, data, label, idx):
    if data is None:
        data = whole_data[idx, :, :]
        label = whole_labels[idx]
    else:
        data = torch.cat([data, whole_data[idx, :, :]])
        label = torch.cat([label, whole_labels[idx]])

    return data, label


def get_custom_MNIST(train_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], root="./data", download=True, transform=None,
                     batch_size=128, shuffle=True, num_workers=2, from_dataset="both"):
    """ get_custom_MNIST

    Keyword Arguments:
    ------------------
        train_labels {list} -- label for training (default: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        root {str} -- root folder name of dataset (default: "./data")
        download {bool} -- whether download dataset or not (default: True)
        transform {torchvision.transforms} -- transform (default: None)
        batch_size {int} -- batch size (default: 128)
        shuffle {bool} -- whether shuffle dataset (default: True)
        num_workers {int} -- num workers (default: 2)
        from_dataset {str} -- where you want to extract data
                              boto: from training and testing dataset
                              train: from training dataset
                              test: from testing dataset

    Returns:
    --------
        {torch.utils.data.DataLoader} -- data loader for train
        {torch.utils.data.DataLoader} -- data loader for test
    """
    test_labels = [i for i in range(10) if not i in train_labels]
    train_dataset = _generate_dataset(datasets=__DATASET["MNIST"], root=root, download=download, train=True, transform=transform)
    test_dataset = _generate_dataset(datasets=__DATASET["MNIST"], root=root, download=download, train=False, transform=transform)
    whole_data = None
    whole_labels = None

    if from_dataset == "both":
        whole_data = train_dataset.data
        whole_data = torch.cat([whole_data, test_dataset.data])
        whole_labels = train_dataset.targets
        whole_labels = torch.cat([whole_labels, test_dataset.targets])
    elif from_dataset == "train":
        whole_data = train_dataset.data
        whole_labels = train_dataset.targets
    else:
        whole_data = test_dataset.data
        whole_labels = test_dataset.targets

    train_data, train_label = None, None
    test_data, test_label = None, None

    for tlabel in train_labels:
        idx = (whole_labels == tlabel)
        train_data, train_label = _append_data(whole_data, whole_labels, train_data, train_label, idx)

    for tlabel in test_labels:
        idx = (whole_labels[:] == tlabel)
        test_data, test_label = _append_data(whole_data, whole_labels,  test_data, test_label, idx)

    train_dataset = CustomDataset(train_data, train_label, transform)
    test_dataset = CustomDataset(test_data, test_label, transform)
    train_dataloader = generate_dataloader(train_dataset, batch_size, shuffle, num_workers)
    test_dataloader = generate_dataloader(test_dataset, batch_size, shuffle, num_workers)
    print(train_dataloader.dataset.size(), test_dataloader.dataset.size())
    return train_dataloader, test_dataloader


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