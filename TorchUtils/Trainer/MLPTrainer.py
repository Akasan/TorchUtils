import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ._Base import TrainerBase
from ..Core.EnvironmentChecker import get_device_type


def calculate_accuracy(outputs, labels):
    return (outputs.max(1)[1] == labels).sum().item()


class MLPClassificationTrainer(TrainerBase):
    def __init__(self, model, criterion, optimizer, device=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        if not device is None:
            self.device = device
        else:
            self.device = get_device_type()

        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    def fit(self, train_loader, epochs, reshape_size=None, verbose_rate=1, validation_loader=None):
        for epoch in range(epochs):
            train_loss = 0.0
            train_acc = 0.0

            self.model.train()

            for images, labels in train_loader:
                if type(reshape_size) == tuple:
                    images = images.view(*reshape_size)

                images = images.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_acc += calculate_accuracy(outputs, labels)
                train_loss += loss.item()

            train_loss /= len(train_loader.dataset)
            train_acc /= len(train_loader.dataset)
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)

            if not validation_loader is None:
                val_loss = 0.0
                val_acc = 0.0

                self.model.eval()
                with torch.no_grad():
                    for images, labels in validation_loader:
                        if type(reshape_size) == tuple:
                            images = images.view(*reshape_size)

                        images = images.to(self.device)
                        outputs = self.model(images)
                        val_acc += calculate_accuracy(outputs, labels)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()

                    val_loss /= len(validation_loader.dataset)
                    val_acc /= len(validation_loader.dataset)
                    self.val_loss_history.append(val_loss)
                    self.val_acc_history.append(val_acc)

            if epoch % verbose_rate == 0:
                print(f"Epoch [{epoch+1} / {epochs}] accuracy: {train_acc} loss: {train_loss}")

    def predict(self, test_loader, reshape_size=None):
        for images, labels in test_loader:
            if type(reshape_size) == tuple:
                images = images.view(*reshape_size)

            images = images.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

    def save(self, model_path="model.pth"):
        torch.save(self.model.state_dict(), model_path)

    def read(self, model_path="model.pth"):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def plot_result(self):
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.train_acc_history, label="train", color="r")
        plt.plot(self.val_acc_history, label="val", color="b")
        plt.xlabel("epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.train_loss_history, label="train", color="r")
        plt.plot(self.val_loss_history, label="val", color="b")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


class MLPAutoEncoderTrainer(TrainerBase):
    def __init__(self, model, criterion, optimizer, device=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        if not device is None:
            self.device = device
        else:
            self.device = get_device_type()

        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    def fit(self, train_loader, epochs, reshape_size=None, verbose_rate=1, validation_loader=None):
        for epoch in range(epochs):
            train_loss = 0.0
            train_acc = 0.0

            self.model.train()

            for images, labels in train_loader:
                if type(reshape_size) == tuple:
                    images = images.view(*reshape_size)

                images = images.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, outputs)
                loss.backward()
                self.optimizer.step()

                train_acc += calculate_accuracy(outputs, labels)
                train_loss += loss.item()

            train_loss /= len(train_loader.dataset)
            train_acc /= len(train_loader.dataset)
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)

            if not validation_loader is None:
                val_loss = 0.0
                val_acc = 0.0

                self.model.eval()
                with torch.no_grad():
                    for images, labels in validation_loader:
                        if type(reshape_size) == tuple:
                            images = images.view(*reshape_size)

                        images = images.to(self.device)
                        outputs = self.model(images)
                        val_acc += calculate_accuracy(outputs, labels)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()

                    val_loss /= len(validation_loader.dataset)
                    val_acc /= len(validation_loader.dataset)
                    self.val_loss_history.append(val_loss)
                    self.val_acc_history.append(val_acc)

            if epoch % verbose_rate == 0:
                print(f"Epoch [{epoch+1} / {epochs}] accuracy: {train_acc} loss: {train_loss}")

    def predict(self, test_loader, reshape_size=None):
        for images, labels in test_loader:
            if type(reshape_size) == tuple:
                images = images.view(*reshape_size)

            images = images.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

    def save(self, model_path="model.pth"):
        torch.save(self.model.state_dict(), model_path)

    def read(self, model_path="model.pth"):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def plot_result(self):
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.train_acc_history, label="train", color="r")
        plt.plot(self.val_acc_history, label="val", color="b")
        plt.xlabel("epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.train_loss_history, label="train", color="r")
        plt.plot(self.val_loss_history, label="val", color="b")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()