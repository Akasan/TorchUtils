import sys
sys.path.append("../")
from TorchUtils.Core.ShapeChecker import check_shape
import torch
import torch.nn as nn



if __name__ == "__main__":
    class Test(nn.Module):
        def __init__(self):
            super(Test, self).__init__()
            self.model = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3),
                nn.Conv2d(64, 64, kernel_size=3),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 128, kernel_size=3),
                nn.Conv2d(128, 128, kernel_size=3),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.Conv2d(256, 256, kernel_size=3),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(256, 512, kernel_size=3),
                nn.Conv2d(512, 512, kernel_size=3),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(412, 1024, kernel_size=3),
                nn.Conv2d(1024, 1024, kernel_size=3),
                nn.Upsample(scale_factor=2),
                nn.Dropout(0.5),
                nn.BatchNorm2d(1024)
            )

        def forward(self, x):
            return self.conv1(x)

    model = Test()
    check_shape(model, (572, 572, 1), is_no_shape_check=True)
