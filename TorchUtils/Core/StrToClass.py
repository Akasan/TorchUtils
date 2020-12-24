from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


ACTIVATION: Dict[str, Any] = {
    "relu": nn.ReLU(True),
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax()
}