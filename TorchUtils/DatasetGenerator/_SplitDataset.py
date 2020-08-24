import torch


def split_dataset(dataset, validation_rate):
    n_samples = len(dataset)
    val_size = int(n_samples * validation_rate)
    train_size = n_samples - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               [train_size, val_size])
    return train_dataset, val_dataset
