from math import ceil, floor
from pprint import pprint
import torch
import torch.nn as nn

from core_class import get_type


class InvalidShapeError(Exception):
    def __init__(self, mes=""):
        super(InvalidShapeError, self).__init__(mes)


def check_shape(model, input_shape, output_shape=None, is_no_shape_check=False):
    shape_history = {}

    for i, layer in enumerate(model.named_modules()):
        if i == 0:
            _shape = "1d" if len(input_shape)  == 1 else "2d"
            shape_history[i] = {"in": input_shape, "out": input_shape, "type": "Dummy", "shape": _shape}
            continue

        _layer = layer[1]
        layer_type = get_type(_layer)

        if layer_type == "linear":
            if not _is_available_shape(shape_history[i-1]["out"], _layer.in_features, shape_history[i-1]["shape"], "1d") and not is_no_shape_check:
                raise InvalidShapeError(f"Specified model is invalid.")

            shape_history[i] = {"in": _layer.in_features, "out": _layer.out_features, "type": type(_layer), "shape": "1d"}
            pass

        elif layer_type == "conv":
            if not _is_available_shape(shape_history[i-1]["out"], _layer.in_channels, shape_history[i-1]["shape"], "2d") and not is_no_shape_check:
                raise InvalidShapeError(f"Specified model is invalid.")

            _output_shape = _calculate_convolutional_output_shape(shape_history[i-1]["out"],
                                                                  _layer.out_channels,
                                                                  _layer.kernel_size,
                                                                  _layer.padding,
                                                                  _layer.stride)

            shape_history[i] = {"in": shape_history[i-1]["out"], "out": _output_shape, "type": type(_layer), "shape": "2d"}

        elif layer_type == "mpool":
            _output_shape = _calculate_pooling_output_shape(shape_history[i-1]["out"],
                                                            _layer.kernel_size,
                                                            _layer.padding,
                                                            _layer.stride,
                                                            _layer.dilation)

            shape_history[i] = {"in": shape_history[i-1]["out"], "out": _output_shape, "type": type(_layer), "shape": "2d"}

        elif layer_type == "upsample":
            # if not _is_available_shape(shape_history[i-1]["out"], _layer.in_channels, shape_history[i-1]["shape"], "2d") and not is_no_shape_check:
            #     raise InvalidShapeError(f"Specified model is invalid.")

            # _output_shape = _calculate_upsample_shape(shape_history[i-1]["out"],
            #                                           _layer.out_channels,
            #                                           _layer.kernel_size,
            #                                           _layer.stride,
            #                                           _layer.padding,
            #                                           _layer.output_padding,
            #                                           _layer.dilation)

            print(dir(_layer))
            _output_shape = _calculate_upsample_shape(shape_history[i-1]["out"], _layer.scale_factor)

            shape_history[i] = {"in": shape_history[i-1]["out"], "out": _output_shape, "type": type(_layer), "shape": "2d"}

        else:
            shape_history[i] = {"in": input_shape, "out": input_shape, "type": type(_layer), "shape": shape_history[i-1]["shape"]}

    pprint(shape_history)

    if not output_shape is None:
        is_exact_output_shape = _is_exact_output_shape(output_shape, shape_history[i])
        print(is_exact_output_shape)


def _is_available_shape(previous_output, current_input, previous_output_shape, current_input_shape):
    if previous_output_shape == "2d" and current_input_shape == "2d":
        return True if previous_output[-1] == current_input else False

    elif previous_output_shape == "2d" and current_input_shape == "1d":
        print(previous_output, current_input, previous_output[-1] == current_input)
        previous_all_cell = previous_output[0] * previous_output[1] * previous_output[2]
        return True if previous_all_cell == current_input else False

    elif previous_output_shape == "1d" and current_input_shape == "1d":
        return True if previous_output == current_input else False

    return False


def _calculate_convolutional_output_shape(input_shape, chennels, kernel_size, padding=0, stride=1):
    output_height = ceil((input_shape[0] + 2*padding[0] - kernel_size[0]) / stride[0] + 1)
    output_width = ceil((input_shape[1] + 2*padding[1] - kernel_size[1]) / stride[1] + 1)
    return (output_height, output_width, chennels)


def _calculate_pooling_output_shape(input_shape, kernel_size, padding, stride, dilation):
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


# TODO nn.Upsampleに変更する
# def _calculate_upsample_shape(input_shape, channels, kernel_size, stride, padding, output_padding, dilation):
#     if type(kernel_size) == int:
#         kernel_size = [kernel_size, kernel_size]

#     if type(padding) == int:
#         padding = [padding, padding]

#     if type(stride) == int:
#         stride = [stride, stride]

#     if type(dilation) == int:
#         dilation = [dilation, dilation]

#     if type(output_padding) == int:
#         output_padding = [output_padding, output_padding]

#     output_height = (input_shape[0] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
#     output_width = (input_shape[1] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1
#     return (output_height, output_width, channels)

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
                nn.Upsample(scale_factor=2)
            )

        def forward(self, x):
            return self.conv1(x)

    model = Test()
    check_shape(model, (572, 572, 1), is_no_shape_check=True)