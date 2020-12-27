import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
from typing import Any, Tuple, Union, List, Sequence, Optional

__DEFAULT_TRANSFORM = transforms.Compose([])


class DataAugmentationGenerator:
    def __init__(self, transform: transforms.Compose = __DEFAULT_TRANSFORM):
        self.transform = transform

    def print_summary(self):
        print(self.transform)

    def get_transform(self) -> torchvision.transforms.Compose:
        return self.transform

    def delete(self, idx: int):
        """ delete

        Arguments:
        ----------
            idx {int} -- order which you want to delete transform
        """
        del self.transform.transforms[idx]

    def _append(self, transform: Any, is_random_apply: bool = False,
                random_apply_rate: float = 0.5):
        if is_random_apply:
            self.transform.transforms.append(transforms.RandomApply([transform], p=random_apply_rate))
        else:
            self.transform.transforms.append(transform)

    def add_tensor(self):
        self._append(transforms.ToTensor())

    def add_resize(self, new_size: int, interpolation: int = 2,
                   is_random_apply: bool = False, random_apply_rate: float = 0.5):
        """ add_resize

        Arguments:
        ----------
            new_size {int or tuple} -- resized size
        """
        self._append(transforms.Resize(new_size, interpolation), is_random_apply, random_apply_rate)

    def add_center_crop(self, crop_size: int, is_random_apply: bool = False,
                        random_apply_rate: float = 0.5):
        """ add_center_crop

        Arguments:
        ----------
            crop_size {int or tuple} -- cropped size
        """
        self._append(transforms.CenterCrop(crop_size), is_random_apply, random_apply_rate)

    def add_color_jitter(self, brightness: float = 0, contrast: Union[float, Tuple[float]] = 0,
                         saturation: Union[float, Tuple[float]] = 0, hue: Union[float, Tuple[float]] = 0,
                         is_random_apply: bool = False, random_apply_rate: float = 0.5, **kwargs):
        self._append(transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, **kwargs),
                     is_random_apply, random_apply_rate)

    def add_pad(self, padding: Union[int, Tuple[int]], fill: Union[int, Tuple[int]] = 0,
                padding_mode: str = "constant", is_random_apply: bool = False,
                random_apply_rate: float = 0.5):
        self._append(transforms.Pad(padding, fill=fill, padding_mode=padding_mode),
                     is_random_apply, random_apply_rate)

    def add_random_affine(self, degree: Union[int, float, Sequence], translate: Union[Tuple, Optional[Any]] = None,
                          scale: Union[Tuple, Optional[Any]] = None, resample: Any = False,
                          shear: Union[Sequence, float, int, Optional[Any]]=None, fillcolor: int = 0,
                          is_random_apply: bool = False, random_apply_rate: float = 0.5):
        self._append(transforms.RandomAffine(degree, translate, scale, shear, resample),
                     is_random_apply, random_apply_rate)

    def add_random_grayscale(self, p: float, is_random_apply: bool = False,
                             random_apply_rate:float = 0.5):
        self._append(transforms.RandomGrayscale(p), is_random_apply, random_apply_rate)

    def add_random_crop(self, crop_size: Union[Sequence, int], is_random_apply: bool = False,
                        random_apply_rate=0.5):
        """ add_random_crop

        Arguments:
        ----------
            crop_size {int or tuple} -- cropped size
        """
        self._append(transforms.RandomCrop(crop_size), is_random_apply, random_apply_rate)

    def add_random_horizontal_flip(self, p: float = 0.5, is_random_apply: bool = False,
                                   random_apply_rate: float = 0.5):
        """ add_random_horizontal_flip

        Keyword Arguments:
        ------------------
            p {float} -- probability (default: 0.5)
        """
        self._append(transforms.RandomHorizontalFlip(p), is_random_apply, random_apply_rate)

    def add_random_vertical_flip(self, p: float = 0.5, is_random_apply: bool = False,
                                 random_apply_rate: float = 0.5):
        """ add_random_vertical_flip

        Keyword Arguments:
        ------------------
            p {float} -- probability (default: 0.5)
        """
        self._append(transforms.RandomVerticalFlip(p), is_random_apply, random_apply_rate)

    def add_random_rotation(self, degree: Union[Sequence, float, int], is_random_apply: bool = False,
                            random_apply_rate: float = 0.5):
        """ add_random_rotation

        Arguments:
        ----------
            degree {int} -- degree
        """
        self._append(transforms.RandomRotation(degrees=degree), is_random_apply, random_apply_rate)

    def add_random_erasing(self, p: float = 0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0,
                           is_random_apply: bool = False, random_apply_rate: float = 0.5):
        """ add_random_erasing

        Keyword Arguments:
        ------------------
            p {float} -- [description] (default: 0.5)
            scale {tuple} -- [description] (default: (0.02, 0.33))
            ratio {tuple} -- [description] (default: (0.3, 3.3))
            value {int} -- [description] (default: 0)
        """
        self._append(transforms.RandomErasing(p=p, scale=scale, ratio=ratio, value=value),
                     is_random_apply, random_apply_rate)

    def add_linear_transformation(self, transformation_matrix, mean_vector,
                                  is_random_apply: bool = False, random_apply_rate: float = 0.5):
        self._append(transforms.LinearTransformation(transformation_matrix, mean_vector), is_random_apply, random_apply_rate)

    def add_normalize(self, mean: Union[Tuple[float], float], std: Union[Tuple[float], float],
                      is_random_apply: bool = False, random_apply_rate: float = 0.5):
        mean = mean if isinstance(mean, tuple) else tuple([mean])
        std = std if isinstance(std, tuple) else tuple([std])
        self._append(transforms.Normalize(mean, std), is_random_apply, random_apply_rate)