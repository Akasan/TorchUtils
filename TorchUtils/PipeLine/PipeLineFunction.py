import numpy as np
import torch


def numpy_to_tensor(ndarray):
    return torch.Tensor(ndarray)


def tensor_to_numpy(tensor):
    return tensor.detach().numpy()
