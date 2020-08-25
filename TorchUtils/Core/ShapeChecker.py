from math import ceil, floor
from pprint import pprint
import torch
import torch.nn as nn

from .TypeChecker import get_type
from .Errors import *


def check_shape(model, input_shape, output_shape=None, is_no_shape_check=False, is_print=True):
    """ check_shape

    Arguments:
    ----------
        model {} -- model you want to check shape
        input_shape {tuple(int)} -- input shape

    Keyword Arguments:
    ------------------
        output_shape {tuple(int)} -- output shape (default: None)
        is_no_shape_check {bool} -- True when you don't want to check the in-out shape (default: False)

    Raises:
    -------
        InvalidShapeError: this raises when the shape is invalid.

    Examples:
    ---------
        >>>
    """
    shape_history = {}

    for i, layer in enumerate(model.named_modules()):
        if i == 0:
            if type(input_shape) == int:
                _shape = "1d"
            else:
                _shape = "2d"
            shape_history[i] = {"in": input_shape, "out": input_shape, "layer": "Dummy", "shape": _shape}
            continue

        _layer = layer[1]
        layer_type = get_type(_layer, as_string=False)

        if layer_type == nn.Linear:
            if not _is_available_shape(shape_history[i-1]["out"], _layer.in_features, shape_history[i-1]["shape"], "1d") and not is_no_shape_check:
                raise InvalidShapeError(f"Specified model is invalid.")

            shape_history[i] = {"in": _layer.in_features, "out": _layer.out_features, "layer": "Linear", "shape": "1d"}

        elif "activation" in str(layer_type):
            activation_type = str(_layer).split(".")[-1].split("'")[0].split("(")[0]
            shape_history[i] = {"in": shape_history[i-1]["out"], "out": shape_history[i-1]["out"], "shape": shape_history[i-1]["shape"], "layer": activation_type}

        elif layer_type == nn.Conv2d:
            if not _is_available_shape(shape_history[i-1]["out"], _layer.in_channels, shape_history[i-1]["shape"], "2d") and not is_no_shape_check:
                raise InvalidShapeError(f"Specified model is invalid.")

            _output_shape = _calculate_convolutional_output_shape(shape_history[i-1]["out"],
                                                                  _layer.out_channels,
                                                                  _layer.kernel_size,
                                                                  _layer.padding,
                                                                  _layer.stride)

            shape_history[i] = {"in": shape_history[i-1]["out"], "out": _output_shape, "layer": "Convolution 2d", "shape": "2d"}

        elif layer_type == nn.MaxPool2d:
            _output_shape = _calculate_pooling_output_shape(shape_history[i-1]["out"],
                                                            _layer.kernel_size,
                                                            _layer.padding,
                                                            _layer.stride,
                                                            _layer.dilation)

            shape_history[i] = {"in": shape_history[i-1]["out"], "out": _output_shape, "layer": "Max Pooling 2d", "shape": "2d"}

        elif layer_type == nn.Upsample:
            _output_shape = _calculate_upsample_shape(shape_history[i-1]["out"], _layer.scale_factor)
            shape_history[i] = {"in": shape_history[i-1]["out"], "out": _output_shape, "layer": "Upsampling", "shape": "2d"}

        elif layer_type in (nn.BatchNorm1d, nn.BatchNorm2d):
            _shape = "1d" if "1d" in str(layer_type) else "2d"
            shape_history[i] = {"in": shape_history[i-1]["out"], "out": shape_history[i-1]["out"], "layer": f"Batch Normalization {_shape}", "shape": _shape}

        elif layer_type == nn.Dropout2d:
            shape_history[i] = {"in": shape_history[i-1]["out"], "out": shape_history[i-1]["out"], "layer": "Dropout 2d", "shape": "2d"}

        else:
            shape_history[i] = {"in": shape_history[i-1]["out"], "out": shape_history[i-1]["out"], "layer": layer_type, "shape": shape_history[i-1]["shape"]}

    del shape_history[0]

    if is_print:
        pprint(shape_history)

    if not output_shape is None:
        is_exact_output_shape = _is_exact_output_shape(output_shape, shape_history[i])


def _is_available_shape(previous_output, current_input, previous_output_shape, current_input_shape):
    """ _is_available_shape

    Arguments:
    ----------
        previous_output {tuple(int)} -- previous layer's output shape
        current_input {tuple(int)} -- current layer's input shape
        previous_output_shape {str} -- dimension of precious layer
        current_input_shape {str} -- dimension of current layer

    Returns:
    --------
        {bool} -- True when specified layers' relationship acccording to the in-out shape is correct
    """
    if previous_output_shape == "2d" and current_input_shape == "2d":
        return True if previous_output[-1] == current_input else False

    elif previous_output_shape == "2d" and current_input_shape == "1d":
        previous_all_cell = previous_output[0] * previous_output[1] * previous_output[2]
        return True if previous_all_cell == current_input else False

    elif previous_output_shape == "1d" and current_input_shape == "1d":
        return True if previous_output == current_input else False

    return False


def _calculate_convolutional_output_shape(input_shape, output_chennels, kernel_size, padding=(0, 0), stride=(1, 1)):
    """ _calculate_convolutional_output_shape

    Arguments:
    ----------
        input_shape {tuple(int)} -- input shape
        chennels {int} -- output channels
        kernel_size {int, tuple(int)} -- kernel size

    Keyword Arguments:
    ------------------
        padding {int} -- padding (default: 0)
        stride {int} -- stride (default: 1)

    Returns:
    --------
        {tuple(int)} -- output shape
    """
    output_height = ceil((input_shape[0] + 2*padding[0] - kernel_size[0]) / stride[0] + 1)
    output_width = ceil((input_shape[1] + 2*padding[1] - kernel_size[1]) / stride[1] + 1)
    return (output_height, output_width, output_chennels)


def _calculate_pooling_output_shape(input_shape, kernel_size, padding, stride, dilation):
    """ _calculate_pooling_output_shape

    Arguments:
    ----------
        input_shape {tuple(int)} -- input shape
        kernel_size {int, tuple(int)} -- kernel size
        padding {int} -- padiding
        stride {int} -- stride
        dilation {int} -- dilation

    Returns:
    --------
        {tuple(int)} -- output shape
    """
    to_list = lambda x: [x, x]

    if type(kernel_size) == int:
        kernel_size = [kernel_size, kernel_size]

    if type(padding) == int:
        padding = [padding, padding]

    if type(stride) == int:
        stride = [stride, stride]

    if type(dilation) == int:
        dilation = [dilation, dilation]

    output_height = floor((input_shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] -1) - 1) / stride[0] + 1)
    output_width = floor((input_shape[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] -1) - 1) / stride[1] + 1)
    return (output_height, output_width, input_shape[-1])


def _calculate_upsample_shape(input_shape, scale_factor):
    return (input_shape[0]*scale_factor, input_shape[1]*scale_factor, input_shape[-1])


def _is_exact_output_shape(exact_shape, model_shape):
    return True if exact_shape == model_shape["out"] else False



if __name__ == "__main__":
    class Test(nn.Module):
        def __init__(self):
            super(Test, self).__init__()
            self.model = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3),
                nn.Conv2d(64, 64, kernel_size=3),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 128, kernel_size=3),
                nn.Conv2d(128, 128, kernel_size=3),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.Conv2d(256, 256, kernel_size=3),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(256, 512, kernel_size=3),
                nn.Conv2d(512, 512, kernel_size=3),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(412, 1024, kernel_size=3),
                nn.Conv2d(1024, 1024, kernel_size=3),
                nn.Upsample(scale_factor=2),
            )

        def forward(self, x):
            return self.conv1(x)

    model = Test()
    check_shape(model, (572, 572, 1), is_no_shape_check=True)
