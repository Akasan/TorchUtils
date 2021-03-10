# https://github.com/dkumazaw/onecyclelr/blob/master/onecyclelr.py

from ..Core.EnvironmentChecker import get_device_type
from torch.optim.lr_scheduler import _LRScheduler
from typing import Any
import torch


class LinearLRScheduler(_LRScheduler):
    def __init__(
        self, optimizer: Any, end_lr: float, num_iter: int, last_epoch: int = -1
    ):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(LinearLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        pass


class LRRangeTester:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim,
        criterion: Any,
        device: str = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = get_device_type() if device is None else device
        self.model.to(self.device)
