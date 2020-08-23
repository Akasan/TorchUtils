import sys
sys.path.append("../")
from TorchUtils.DatasetGenerator.FromPublicDatasets import load_public_datasets
from TorchUtils.Trainer import MLPAutoEncoderTrainer
from TorchUtils.ModelGenerator.MLP import MLP
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28**2, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 28**2),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader, test_loader = load_public_datasets("MNIST", transform=transform)
    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.MSELoss()
    trainer = MLPAutoEncoderTrainer(model, criterion, optimizer)
    trainer.fit(train_loader, epochs=5, reshape_size=(-1, 28**2), validation_loader=test_loader)
    trainer.plot_result()