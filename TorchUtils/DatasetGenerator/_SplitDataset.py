import torch
from typing import Tuple


def split_dataset(dataset: torch.utils.data.Dataset,
                  validation_rate: float) ->Tuple[torch.utils.data.Dataset]:
    """ split_dataset to training dataset and validation dataset

    Arguments:
    ----------
        dataset {torch.utils.data.Dataset} -- dataset
        validation_rate {float} -- the rate of validation

    Returns:
    --------
        {torch.utils.data.Dataset} -- training dataset
        {torch.utils.data.Dataset} -- validation dataset
    """
    if validation_rate in (None, 0.0):
        return dataset, None

    n_samples = len(dataset)
    val_size = int(n_samples * validation_rate)
    train_size = n_samples - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset
