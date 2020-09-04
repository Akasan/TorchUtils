import sys
sys.path.append("../")
from TorchUtils.DatasetGenerator.FromPublicDatasets import load_public_dataset
from TorchUtils.Trainer.MLPTrainer import MLPClassificationTrainer
from TorchUtils.ModelGenerator.MLP import MLP
from TorchUtils.Core.ShapeChecker import check_shape
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch_lr_finder import LRFinder


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader, test_loader = load_public_dataset("MNIST", transform=transform)
    model = Model().half()
    optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    # lr_finder = LRFinder(model, optimizer, criterion, device="cpu")
    # lr_finder.range_test(train_loader, end_lr=100, num_iter=100, accumulation_steps=1)
    # lr_finder.plot()
    # lr_finder.reset()
    trainer = MLPClassificationTrainer(model, criterion, optimizer)
    trainer.fit(train_loader, epochs=10, reshape_size=(-1, 28**2), validation_loader=test_loader)
    trainer.plot_result()
