import torch


def get_device_type():
    return "cuda" if torch.cuda.is_available() else "cpu"


def convert_device(*items, device=None):
    result = []
    device = get_device_type() if device is None else device
    for item in items:
        result.append(item.to(device))

    return result