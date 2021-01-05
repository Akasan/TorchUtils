from torch import Tensor
from torch import nn
from typing import List, Any


def calculate_listed_layer(layers: List[Any], x: Tensor) -> Tensor:
    for layer in layers:
        x = layer(x)

    return x
