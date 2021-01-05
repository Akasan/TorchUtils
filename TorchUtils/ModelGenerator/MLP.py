import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..Core.StrToClass import ACTIVATION
from ..Core.ShapeChecker import check_shape


# TODO 直接レイヤーオブジェクトを入れてもいいようにする


class MLP(nn.Module):
    def __init__(self, neurons, activations):
        super(MLP, self).__init__()
        modules = [nn.Linear(neurons[0], neurons[1])]
        if not activations[0] is None:
            modules.append(ACTIVATION[activations[0]])

        for i in range(1, len(neurons)-1):
            modules.append(nn.Linear(neurons[i], neurons[i+1]))
            if not activations[i] is None:
                modules.append(ACTIVATION[activations[i]])

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    neurons = [100, 75, 50, 25, 10]
    activations = [None, "sigmoid", "relu", "softmax"]
    model = MLP(neurons, activations)
    check_shape(model, (100,))