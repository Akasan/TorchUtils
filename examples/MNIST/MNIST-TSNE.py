# Sample of metric learning with CNN and TSNE
import sys
sys.path.append("../..")
from TorchUtils.DatasetGenerator.FromPublicDatasets import load_public_dataset
from TorchUtils.Trainer.CNNTrainer import CNNClassificationTrainer
from TorchUtils.ModelGenerator.MLP import MLP
from TorchUtils.Core.ShapeChecker import check_shape
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch_lr_finder import LRFinder
from sklearn.manifold import TSNE



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(13*13*16, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader, test_loader = load_public_dataset("MNIST", transform=transform, batch_size=256)
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    # lr_finder = LRFinder(model, optimizer, criterion, device="cpu")
    # lr_finder.range_test(train_loader, end_lr=100, num_iter=100, accumulation_steps=1)
    # lr_finder.plot()
    trainer = CNNClassificationTrainer(model, criterion, optimizer)
    trainer.fit(train_loader, epochs=20, validation_loader=test_loader)
    trainer.plot_result()
    trainer.save(is_parameter_only=False)