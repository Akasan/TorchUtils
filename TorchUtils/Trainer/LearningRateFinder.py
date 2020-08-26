# https://github.com/dkumazaw/onecyclelr/blob/master/onecyclelr.py

from ..Core.EnvironmentChecker import get_device_type
from torch.optim.lr_scheduler import _LRScheduler


class LinearLRScheduler(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(LinearLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        pass


class LRRangeTester:
    def __init__(self, model, optimizer, criterion, device=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = get_device_type() if device is None else device
        self.model.to(self.device)

    def execute_range_test(self, train_loader, start_lr=1e-7, end_lr=1e2, num_iter=100):
        self.loss_history = {"lr": [], "loss": []}