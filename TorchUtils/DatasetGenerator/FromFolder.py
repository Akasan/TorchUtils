import torch
import numpy as np
import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from .LoaderGenerator import generate_dataloader


def _generate_dataset(folder_path, transform):
    return ImageFolder(folder_path, transform)


def generate_dataset_loader(folder_path, transform=None, batch_size=128, shuffle=True, num_workers=2):
    dataset = _generate_dataset(folder_path, transform)
    loader = generate_dataloader(dataset, batch_size, shuffle, num_workers)
    return loader


def to_cv2_format(img):
    pass


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    loader = generate_dataset_loader("C:\\Users\\chino\\Downloads\\31797_40972_bundle_archive\\train", transform, batch_size=1)
    import cv2
    # for i, (image, label) in enumerate(loader):
    #     img = image.detach().numpy()*255
    #     img = img.reshape((img.shape[-2], img.shape[-1], 3))
    #     print(type(img), img.shape)
    #     cv2.imshow(f"{label}", img.astype(np.uint8))
    #     while True:
    #         if cv2.waitKey(20) == ord("q"):
    #             cv2.destroyAllWindows()
    #             break
