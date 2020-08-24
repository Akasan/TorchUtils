import sys
sys.path.append("../")
from TorchUtils.Visualizer.Visualizer import Visualizer
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(3, 16, (3, 3))
        )

    def forward(self, x):
        return self.model(x)


model = Model()
visualizer = Visualizer(model)
visualizer.describe_kernel()
