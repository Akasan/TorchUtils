from typing import Any
import torch
import torch.nn as nn
from typing import Any, Dict


def is_tensor(x: Any) -> bool:
    """ check whether specified value x is torch.tesnsor or not

    Arguments:
    ----------
        x {Any} -- value you want to check the type

    Returns:
    --------
        {bool} -- True when input value is torch.Tensor
    """
    return type(x) == torch.Tensor


__TYPE: Dict[Any, str] = {
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

def get_type(layer: Any, as_string: bool = True) -> str:
    """ get type of layer

    Arguments:
    ----------
        layer {torch.nn.modules.*} -- torch layer

    Returns:
    --------
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
