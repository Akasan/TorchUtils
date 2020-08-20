import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self, train_loader, epochs, batch_size):
        for epoch in range(epochs):
            for i, (images, labels) in enumerate(train_loader):
                pass

    def predict(self, test_loader):
        pass

    def save(self, filename):
        pass
