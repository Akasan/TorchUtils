# https://github.com/dkumazaw/onecyclelr/blob/master/onecyclelr.py
class OneCycleChecker:
    def __init__(self, optimizer, steps, lr_range=(1e-7, 1e-1), momentum_range=(0.01, 0.9), annihilation=0.1, reduce_factor=0.01,last_step=-1):
        self.optimizer = optimizer
        self.steps = steps
        self.lr_min, self.lr_max = lr_range
        self.momentum_min, self.momentum_max = momentum_range
        self.cycle_steps = int(steps * (1 - annihilation))
        self.final_lr = self.lr_min * reduce_factor
        self.last_step = last_step