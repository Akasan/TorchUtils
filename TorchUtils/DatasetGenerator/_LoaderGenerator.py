import torch
import torchvision
import warnings
warnings.simplefilter("ignore")


def generate_dataloader(dataset: torch.utils.data.Dataset, batch_size: int,
                        shuffle: bool, num_workers: int) -> torch.utils.data.DataLoader:
    """ generate_dataloader

    Arguments:
    ----------
        dataset {torch.utils.data.Dataset} -- dataset
        batch_size {int} -- batch size
        shuffle {bool} -- whether shuffle dataset or not
        num_workers {int} -- num workers

    Returns:
    --------
        {torch.utils.data.DataLoader} -- data loader
    """
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers)