import torch
import torchvision

def generate_dataloader(dataset, batch_size, shuffle, num_workers):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers)