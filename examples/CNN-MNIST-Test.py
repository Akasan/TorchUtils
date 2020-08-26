import sys
sys.path.append("../")
from TorchUtils.DatasetGenerator.FromPublicDatasets import load_public_dataset
from TorchUtils.Trainer.CNNTrainer import CNNClassificationTrainer
from TorchUtils.ModelGenerator.MLP import MLP
from TorchUtils.Core.ShapeChecker import check_shape
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch_lr_finder import LRFinder


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(True),
            # nn.MaxPool2d(32, 26)
        )

        self.classification = nn.Sequential(
            nn.Linear(32, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.model(x)
        x = F.max_pool2d(x, kernel_size=26)
        x = x.view(x.size()[0], 32)
        return self.classification(x)


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader, test_loader = load_public_dataset("MNIST", transform=transform)
    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    lr_finder = LRFinder(model, optimizer, criterion, device="cpu")
    lr_finder.range_test(train_loader, end_lr=10, num_iter=1000)
    lr_finder.plot()
    input()
    trainer = CNNClassificationTrainer(model, criterion, optimizer)
    # check_shape(model, (28, 28, 1))
    # input()
    trainer.fit(train_loader, epochs=5, reshape_size=(-1, 1, 28, 28), validation_loader=test_loader)
    trainer.plot_result()