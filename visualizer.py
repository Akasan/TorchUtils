import numpy as np
from pprint import pprint
from core_class import is_tensor, get_type
from error_class import *
import cv2
from imglib.image_io import imshow_mul
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    def __init__(self, model):
        self.__model = model
        self._separate_layers()

    def _separate_layers(self):
        """ separate model's layer"""
        self.__layer = {}

        for i, layer in enumerate(self.__model.named_modules()):
            if i == 0:
                continue

            self.__layer[layer[0]] = layer[1]

    def describe_layers(self, layer_name=None, row=1, column=1):
        """ describe each layer's information

        Arguments:
            layer_name {str} -- if you want to describe speficy layer, set this as the layer name
                                (default: None)
            row {int} -- the number of rows when you want to show kernel as images(default: 1)
            coulmn {int} -- the number of columns when you want to show kernel as images(default: 1)
        """
        if layer_name is None:
            pprint(self.__layer)
        else:
            if get_type(self.__layer[layer_name]) == "conv":
                kernel = self.__layer[layer_name].weight
                self._generate_heatmap(kernel, row, column)

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

        self._generate_output_image(inputs, self.__layer[layer_name].weight[0])
        outputs_orig = self.__layer[layer_name](inputs)

        if is_show:
            inputs = np.array(inputs).astype(np.uint8) * 255
            outputs = np.array(outputs_orig).astype(np.uint8) * 255
            imshow_mul([inputs, outputs], ["inputs", "outputs"])

        return outputs_orig

    def _generate_heatmap(self, kernel, row, column, is_gray=True):
        """ generate heatmap

        Arguments:
            kernel {torch.tensor} -- kernel of convolution layer
            row {int} -- the number of rows
            column {int} -- the number of columns

        Keyword Arguments:
            is_gray {bool} -- whether generate heatmap as gray scale(default: True)
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