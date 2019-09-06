import torch
import torch.nn as nn


TORCH_TENSOR_TYPE = type(torch.tensor([1]))


def is_tensor(x):
    """ check whether specified value x is torch.tesnsor or not

    Arguments:
        x {any} -- value you want to check the type
    """
    return True if type(x) == TORCH_TENSOR_TYPE else False


def show_images(images, window_name, is_reguralized=True):
    """ show images"""
    pass


__TYPE = {
    nn.Conv2d: "conv",
}

def get_type(layer):
    """ get type of layer

    Arguments:
        layer {torch.nn.modules.*} -- torch layer

    Returns:
        {str} -- layer type as character
    """
    return __TYPE[type(layer)]