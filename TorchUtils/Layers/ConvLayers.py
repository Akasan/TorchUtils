import torch
from torch import nn


class FireModule(nn.Module):
    def __init__(self, input_channels, squeeze_channels, expand_channels):
        """ __init__

        Arguments:
        ----------
            input_channels {int} -- input channel size
            squeeze_channels {int} -- output channel size of squeeze block
            expand_channels {int} -- output channel size of expand block
        """
        super(FireModule, self).__init__()
        # INPUT
        self.squeeze = nn.Sequential(
            nn.Conv2d(input_channels, squeeze_channels, kernel_size=1, stride=1),
            nn.ReLU(True)
        )

        # 1X1 expand block
        self.expand1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1, stride=1),
            nn.ReLU(True)
        )

        # 3X3 expand block
        self.expand3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, stride=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        squeeze_out = self.squeeze(x)
        expand1_out = self.expand1(squeeze_out)
        expand3_out = self.expand3(squeeze_out)
        return torch.cat([expand1_out, expand3_out], 1)