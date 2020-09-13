# Sample of metric learning with CNN and TSNE
import sys
sys.path.append("../..")
from TorchUtils.DatasetGenerator.FromPublicDatasets import load_public_dataset
from TorchUtils.Trainer.CNNTrainer import CNNClassificationTrainer
from TorchUtils.ModelGenerator.MLP import MLP
from TorchUtils.Core.ShapeChecker import check_shape
from TorchUtils.Visualizer.ManifoldVisualizer import TSNEVizualizer, SVCVisualizer, PCAVizualizer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
# from torch_lr_finder import LRFinder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(True),
        )

        self.pre_classifier = nn.Sequential(
            nn.Linear(13*13*16, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x =  self.pre_classifier(x)
        return self.classifier(x)


def predict(model, test_loader, reshape_size=None):
    total_outputs = None
    total_labels = None
    for images, labels in test_loader:
        images = images.to("cpu")
        outputs = model.features(images)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = model.pre_classifier(outputs)
        if total_outputs is None:
            total_outputs = outputs.detach().numpy()
            total_labels = labels.detach().numpy()
            total_labels = total_labels.reshape(total_labels.shape[0], 1)
        else:
            total_outputs = np.vstack((total_outputs, outputs.detach().numpy()))
            labels_temp = labels.detach().numpy()
            total_labels = np.vstack((total_labels, labels_temp.reshape(labels_temp.shape[0], 1)))

    return total_outputs, total_labels


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader, test_loader = load_public_dataset("MNIST", transform=transform, batch_size=256)
    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    trainer = CNNClassificationTrainer(model, criterion, optimizer)
    trainer.fit(train_loader, epochs=20, validation_loader=test_loader)
    outputs, labels = predict(model, test_loader)
    tsne_visualizer = TSNEVizualizer(n_components=2)
    tsne_visualizer.fit_transform(outputs, labels)
    tsne_visualizer.plot()

    # svc_visualizer = PCAVizualizer()
    # svc_visualizer.fit(outputs)
    # svc_visualizer.calculate_stats(outputs, labels)
    # svc_visualizer.plot(outputs, labels)