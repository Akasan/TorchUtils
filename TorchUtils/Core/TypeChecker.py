import torch
import torch.nn as nn
from typing import Any


__TORCH_TENSOR_TYPE = type(torch.tensor([1]))


def is_tensor(x: torch.Tensor) -> bool:
    """ check whether specified value x is torch.tesnsor or not

    Arguments:
        x {any} -- value you want to check the type
    """
    return True if type(x) == __TORCH_TENSOR_TYPE else False


__TYPE = {
    nn.Linear: "linear",

    nn.Conv1d: "conv",
    nn.Conv2d: "conv",
    nn.Conv3d: "conv",

    nn.MaxPool1d: "mpool",
    nn.MaxPool2d: "mpool",
    nn.MaxPool3d: "mpool",

    nn.ConvTranspose2d: "upsample",
    nn.Upsample: "upsample",

    nn.BatchNorm1d: "batchnorm",
    nn.BatchNorm2d: "batchnorm",

    nn.Dropout: "dropout",
}

def get_type(layer: Any, as_string: bool = True) -> Any:
    """ get type of layer

    Arguments:
        layer {torch.nn.modules.*} -- torch layer

    Returns:
        {str} -- layer type as character
    """
    try:
        _type = type(layer)
        if "activation" in str(_type):
            return "activation"

        return __TYPE[_type] if as_string else _type
    except:
        # __TYPEに登録されていないレイヤー
        return type(layer)
