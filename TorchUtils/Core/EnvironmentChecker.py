import torch


def get_device_type():
    return "cuda" if torch.cuda.is_available() else "cpu"
