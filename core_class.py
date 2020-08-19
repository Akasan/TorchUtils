import torch
import torch.nn as nn


__TORCH_TENSOR_TYPE = type(torch.tensor([1]))


def is_tensor(x):
    """ check whether specified value x is torch.tesnsor or not

    Arguments:
        x {any} -- value you want to check the type
    """
    return True if type(x) == __TORCH_TENSOR_TYPE else False


def show_images(images, window_name, is_reguralized=True):
    """ show images"""
    pass


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
}

def get_type(layer):
    """ get type of layer

    Arguments:
        layer {torch.nn.modules.*} -- torch layer

    Returns:
        {str} -- layer type as character
    """
    try:
        _type = __TYPE[type(layer)]
        return _type
    except:
        return type(layer)