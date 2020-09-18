import sys
sys.path.append("../..")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
# from torch_lr_finder import LRFinder
from sklearn.manifold import TSNE

# from TorchUtils.Analyzer.LayerAnalyzer import AnalyzedLinear
from TorchUtils.Analyzer.ManifoldAnalyzer import (LDAAnalyzer, PCAAnalyzer,
                                                  SVCAnalyzer,
                                                  TruncatedSVDAnalyzer,
                                                  TSNEAnalyzer)
from TorchUtils.Core.Arguments import parse_args
from TorchUtils.Core.EnvironmentChecker import get_device_type, convert_device
from TorchUtils.Core.LayerCalculator import calculate_listed_layer
from TorchUtils.Core.ShapeChecker import check_shape
from TorchUtils.DatasetGenerator.FromPublicDatasets import get_custom_MNIST
from TorchUtils.ModelGenerator.MLP import MLP
from TorchUtils.PipeLine.PipeLine import PipeLine
from TorchUtils.Trainer.CNNTrainer import CNNClassificationTrainer
from TorchUtils.Layers.ConvLayers import FireModule
import warnings
warnings.simplefilter("ignore")


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.activation1 = nn.ReLU(True)
        self.features = nn.ModuleList([self.conv1, self.pool1, self.activation1])

        self.linear1 = nn.Linear(13*13*16, 512)
        self.activation2 = nn.ReLU(True)
        self.linear2 = nn.Linear(512, 128)
        self.activation3 = nn.ReLU(True)
        self.pre_classifier = nn.ModuleList([self.linear1, self.activation2, self.linear2, self.activation3])

        # self.linear3 = AnalyzedLinear(128, 10)
        self.linear3 = nn.Linear(128, 10)
        self.activation4 = nn.Softmax()
        self.classifier = nn.ModuleList([self.linear3, self.activation4])


    def forward(self, x):
        x = calculate_listed_layer(self.features, x)
        x = x.view(x.size(0), -1)
        x = calculate_listed_layer(self.pre_classifier, x)
        return calculate_listed_layer(self.classifier, x)


def predict(model, test_loader, reshape_size=None):
    total_outputs = None
    total_labels = None
    for images, labels in test_loader:
        images, labels = convert_device(images, labels)
        outputs = calculate_listed_layer(model.features, images)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = calculate_listed_layer(model.pre_classifier, outputs)

        if total_outputs is None:
            total_outputs = outputs.detach().numpy()
            total_labels = labels.detach().numpy()
            total_labels = total_labels.reshape(total_labels.shape[0], 1)
        else:
            total_outputs = np.vstack((total_outputs, outputs.detach().numpy()))
            labels_temp = labels.detach().numpy()
            total_labels = np.vstack((total_labels, labels_temp.reshape(labels_temp.shape[0], 1)))

    return total_outputs, total_labels


def train_model(args):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader, val_loader, test_loader = get_custom_MNIST(train_labels=[0, 2, 4, 6, 8], transform=transform,
                                                             val_rate=0.2)
    model = Model().to(get_device_type())
    optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    trainer = CNNClassificationTrainer(model, criterion, optimizer)
    trainer.fit(train_loader, epochs=args.epochs, validation_loader=val_loader)
    trainer.evaluate(test_loader)
    # outputs, labels = predict(model, test_loader)
    # return outputs, labels


def train_svd(data):
    svd = TruncatedSVDAnalyzer()
    svd.fit(data[0])
    outputs = svd.predict(data[0])
    return (outputs, data[1])


def train_svm(data):
    svc_visualizer = SVCAnalyzer()
    svc_visualizer.fit(data[0], data[1])
    svc_visualizer.evaluate(data[0], data[1])
    svc_visualizer.plot(data[0], data[1])


def train_lda(data):
    lda_visualizer = LDAAnalyzer()
    lda_visualizer.fit(data[0], data[1])
    lda_visualizer.plot(data[0], data[1])


if __name__ == "__main__":
    args = parse_args()
    pipeline = PipeLine()
    pipeline.add_function(train_model, False, args)
    # pipeline.add_function(train_svd)
    # pipeline.add_function(train_lda)
    # pipeline.add_function(train_svm)
    pipeline.execute()
    pipeline.save_pipeline()
