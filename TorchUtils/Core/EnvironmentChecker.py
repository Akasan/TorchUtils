import torch
import warnings
from typing import Union, List

warnings.simplefilter("ignore")


def get_device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def convert_device(*items, device: Union[str, None] = None) -> List[torch.Tensor]:
    device = get_device_type() if device is None else device
    return [item.to(device) for item in items]
