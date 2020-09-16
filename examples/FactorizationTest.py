import torch
from torch import nn
import sys
sys.path.append("../")
from TorchUtils.Core.ShapeChecker import check_shape
from torchsummary import summary


class Model5(nn.Module):
    def __init__(self):
        super(Model5, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=5)

    def forward(self, x):
        return self.conv1(x)


class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


if __name__ == "__main__":
    model5 = Model5().to("cpu")
    model3 = Model3().to("cpu")
    model = Model().to("cpu")
    input_shape = (1, 100, 100)
    summary(model5, input_shape)
    print("="*30)
    summary(model3, input_shape)
    print("="*30)
    summary(model, input_shape)
