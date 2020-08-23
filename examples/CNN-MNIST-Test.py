import sys
sys.path.append("../")
from TorchUtils.DatasetGenerator.FromPublicDatasets import load_public_datasets
from TorchUtils.Trainer import CNNClassificationTrainer
from TorchUtils.ModelGenerator.MLP import MLP
from TorchUtils.Core.ShapeChecker import check_shape
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(True)
        )

        self.classification = nn.Sequential(
            nn.Linear(21632, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.classification(x)


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader, test_loader = load_public_datasets("MNIST", transform=transform)
    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    trainer = CNNClassificationTrainer(model, criterion, optimizer)
    check_shape(model, (28, 28, 1))
    trainer.fit(train_loader, epochs=5, reshape_size=(-1, 1, 28, 28), validation_loader=test_loader)
    trainer.plot_result()