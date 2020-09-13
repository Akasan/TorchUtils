import torch
import torchvision

def generate_dataloader(dataset, batch_size, shuffle, num_workers):
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