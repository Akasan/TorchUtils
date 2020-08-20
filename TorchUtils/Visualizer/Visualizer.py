import numpy as np
from pprint import pprint
from ..Core.TypeChecker import is_tensor, get_type
from ..Core.Errors import *
import cv2
from imglib.image_io import imshow_mul
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    def __init__(self, model):
        """
        Arguments:
            model {torch.nn.module} -- Torch model
        """
        self.__model = model
        self._separate_layers()

    def _separate_layers(self):
        """ separate model's layer and keep as dictionary"""
        self.__layer = {}

        for i, layer in enumerate(self.__model.named_modules()):
            if i == 0:
                continue

            layer_type = get_type(layer[1])
            self.__layer[layer[0]] = {"type": layer_type, "detail": layer[1]}

    def describe_layers(self, layer_name=None):
        """ describe each layer's information

        Arguments:
            layer_name {str} -- if you want to describe speficy layer, set this as the layer name
                                (default: None)
        """
        if layer_name is None:
            print("All layer description:")
            for k, v in self.__layer.items():
                print(f"    Layer name: {k}")
                print(f"        detail: {v['detail']}\n")

        else:
            assert layer_name in self.__layer, f"layer name: {layer_name} doesn't exist"
            print(f"Layer name: {layer_name}")
            print(f"    detail: {self.__layer[layer_name]['detail']}")

    def describe_kernel(self,
                        layer_name=None,
                        row=1,
                        column=1,
                        idx=[],
                        is_all=False):
        """ describe conv layer's kernel

        Keyword Arguments:
            layer_name {str} -- layer name (default: None)
                                - if layer_name is None, you can show layer name and select
            row {int} -- the number of rows you want to show at once (default: 1)
            column {int} -- the number of columns you want to show at once (default: 1)
            idx {list(int)} -- when you want to describe specified kernel, set this (default: None)
            is_all {bool} -- if you want to describe all kernrel, set this as True
        """
        if layer_name == None:
            pprint(self.__layer)        # TODO convだけ表示
            layer_name = input("Layer name >>> ")

        if not layer_name in self.__layer:
            raise NotExistLayerNameError(f"layer name {layer_name} is not existed")

        kernel = self.__layer[layer_name]["detail"].weight
        self._generate_heatmap(kernel, row, column, idx, is_all)


    def generate_output(self, layer_name, inputs, is_show=False):
        """ generate outpu with specify layer

        Arguments:
            layer_name {str} -- layer name you want to generate output
            inputs {torch.tensor} -- input values

        Keyword Arguments:
            is_show {bool} -- whether show as image (default: False)

        Raises:
            NotExistLayerNameError: [description]
            NotTensorError: [description]

        Returns:
            [type] -- [description]
        """
        if not layer_name in self.__layer.keys():
            raise NotExistLayerNameError(f"You specify not existed layer name: {layer_name}")

        if not is_tensor(inputs):
            raise NotTensorError("You specify not torch.tensor")

        self._generate_output_image(inputs, self.__layer[layer_name]["detail"].weight[0])
        outputs_orig = self.__layer[layer_name]["detail"](inputs)

        if is_show:
            inputs = np.array(inputs).astype(np.uint8) * 255
            outputs = np.array(outputs_orig).astype(np.uint8) * 255
            imshow_mul([inputs, outputs], ["inputs", "outputs"])

        return outputs_orig

    def _generate_heatmap(self,
                          kernel,
                          row,
                          column,
                          idx,
                          is_all,
                          is_gray=True):
        """ generate heatmap

        Arguments:
            kernel {torch.tensor} -- kernel of convolution layer
            row {int} -- the number of rows
            column {int} -- the number of columns
            idx {list(int)} -- index of kernel for describe specified one
            is_all {bool} -- whether generate heatmap of all kernels

        Keyword Arguments:
            is_gray {bool} -- whether generate heatmap as gray scale (default: True)
        """
        fix, ax = plt.subplots(nrows=row, ncols=column,
                               figsize=(kernel.shape[2], kernel.shape[3]))

        kernel = kernel[:row*column, 0, :, :].detach().numpy()
        kernel_list = [k.tolist() for k in kernel]

        for i, k in enumerate(kernel_list):
            if is_gray:
                sns.heatmap(k, ax=ax[i//row, i%column], cmap="gray")
            else:
                sns.heatmap(k, ax=ax[i//row, i%column])

        plt.show()

    def _generate_output_image(self, input_image, kernel):
        kernel = kernel.detach().numpy()[0, :, :]
        input_image = input_image[0].detach().numpy()
        output_img = cv2.filter2D(input_image, -1, kernel)
        plt.subplot(131)
        plt.imshow(input_image)
        plt.subplot(132)
        sns.heatmap(kernel)
        plt.subplot(133)
        plt.imshow(output_img)
        plt.show()
