import sys
sys.path.append("../..")

from tqdm import tqdm
import torch
from torch import nn
from TorchUtils.DatasetGenerator.FromPublicDatasets import get_custom_MNIST
from torchvision import transforms
from torchvision.datasets import MNIST

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    train, test = get_custom_MNIST(train_labels=[0, 1, 2, 3, 4], transform=transform, from_dataset="train")
    print(train, test)
