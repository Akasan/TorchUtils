from torch import nn
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore")


class AnalyzedLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AnalyzedLinear, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        self.INPUT_DIM = input_dim
        self.OUTPUT_DIM = output_dim
        self.outputs = {i: None for i in range(output_dim)}
        self.distribution = {i: {} for i in range(output_dim)}

    def forward(self, x):
        outputs = self.layer(x)
        for i in range(self.OUTPUT_DIM):
            self.outputs[i] = outputs[:, i]

        self._calculate_distribution()
        return self.layer(x)

    def _calculate_distribution(self):
        for i in range(self.OUTPUT_DIM):
            self.distribution[i]["mean"] = self.outputs[i].mean().item()
            self.distribution[i]["std"] = self.outputs[i].std().item()

    def plot_dist(self):
        """ plot_dist"""
        for i in range(self.OUTPUT_DIM):
            plt.subplot(1, self.OUTPUT_DIM, i+1)
            sns.distplot(self.outputs[i].detach().numpy(), kde=True)

        plt.show()


class AnalyzedConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(AnalyzedConv2d, self).__init__()
        self.layer = nn.Conv2d(*args, **kwargs)

    def forward(self, x):
        return self.layer(x)