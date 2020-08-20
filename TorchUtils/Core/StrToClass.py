import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIVATION = {
    "relu": nn.ReLU(True),
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax()
}