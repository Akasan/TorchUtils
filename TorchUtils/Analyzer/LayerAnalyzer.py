import torch
from torch import nn
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict

import warnings
warnings.simplefilter("ignore")


class AnalyzedLinear(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super(AnalyzedLinear, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        self.INPUT_DIM = input_dim
        self.OUTPUT_DIM = output_dim
        self.outputs: Dict[int, torch.Tensor] = None
        self.distribution: Dict[int, Dict[str, float]] = dict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.layer(x)
        self.outputs = {i: outputs[:, i] for i in range(self.OUTPUT_DIM)}
        self._calculate_distribution()
        return self.layer(x)

    def _calculate_distribution(self) -> None:
        self.distribution = {i: {"mean": out.mean().item(), "std": out.std().item()}
                             for i, out in self.outputs.items()}

    def plot_dist(self) -> None:
        """ plot_dist"""
        for i in range(self.OUTPUT_DIM):
            plt.subplot(1, self.OUTPUT_DIM, i+1)
            sns.distplot(self.outputs[i].detach().numpy(), kde=True)

        plt.show()


class AnalyzedConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(AnalyzedConv2d, self).__init__()
        self.layer = nn.Conv2d(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)