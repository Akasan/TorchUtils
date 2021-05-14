import sys
import time
import warnings

import colorama
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from colorama import Fore
from typing import Any, Tuple

from ..Core.EnvironmentChecker import convert_device, get_device_type
from ._KeyboardInterruptHandler import respond_exeption
from ._ModelSaver import save_model
from ._Printer import print_result, show_progressbar, summarize_trainer
from ._TrainerInterface import TrainerBase

warnings.simplefilter("ignore")
colorama.init()


def calculate_accuracy(outputs, labels):
    return (outputs.max(1)[1] == labels).sum().item()


class CNNClassificationTrainer(TrainerBase):
    def __init__(
        self, model: torch.nn.Module, criterion: Any, optimizer: Any, device: str = None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = get_device_type() if device is None else device
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        summarize_trainer(self.model, self.criterion, self.optimizer)

    def reset_history(self):
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int,
        reshape_size: Tuple[int] = None,
        verbose_rate: int = 1,
        validation_loader: torch.utils.data.DataLoader = None,
    ):
        print(Fore.RED + "<<< START TRAINING MODEL >>>" + Fore.WHITE)
        self.reset_history()

        try:
            st = time.time()
            for epoch in range(epochs):
                train_loss = 0.0
                train_acc = 0.0
                val_loss = None
                val_acc = None
                self.model.train()
                st = time.time()

                for i, (images, labels) in enumerate(train_loader, 1):
                    show_progressbar(
                        len(train_loader.dataset) // train_loader.batch_size,
                        i,
                        whole_time=time.time() - st,
                    )
                    images, labels = convert_device(images, labels)
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

                time_diff = time.time() - st

                if not validation_loader is None:
                    val_loss = 0.0
                    val_acc = 0.0

                    self.model.eval()
                    with torch.no_grad():
                        for i, (images, labels) in enumerate(validation_loader, 1):
                            show_progressbar(
                                len(validation_loader.dataset)
                                // validation_loader.batch_size,
                                i,
                                is_training=False,
                            )

                            if type(reshape_size) == tuple:
                                images = images.view(*reshape_size)

                            images, labels = convert_device(images, labels)
                            outputs = self.model(images)
                            loss = self.criterion(outputs, labels)
                            val_acc += calculate_accuracy(outputs, labels)
                            val_loss += loss.item()

                        val_loss /= len(validation_loader.dataset)
                        val_acc /= len(validation_loader.dataset)
                        self.val_loss_history.append(val_loss)
                        self.val_acc_history.append(val_acc)

                if (epoch + 1) % verbose_rate == 0:
                    print_result(
                        epoch,
                        epochs,
                        train_acc,
                        train_loss,
                        val_acc,
                        val_loss,
                        time=time_diff,
                    )

        except KeyboardInterrupt:
            respond_exeption(self.model)

    def predict(
        self,
        test_loader: torch.utils.data.DataLoader,
        reshape_size: Tuple[int] = None,
        to_numpy: bool = False,
    ):
        total_outputs = None
        total_labels = None

        for images, labels in test_loader:
            if type(reshape_size) == tuple:
                images = images.view(*reshape_size)

            images = images.to(self.device)
            outputs = self.model(images)

            if total_outputs is None:
                total_outputs = outputs
                total_labels = labels
            else:
                total_outputs = torch.cat((total_outputs, outputs), 0)
                total_labels = torch.cat((total_labels, labels), 0)

        if to_numpy:
            return total_outputs.detach().numpy(), total_labels.detach().numpy()
        else:
            return total_outputs, total_labels

    def evaluate(self, test_loader: torch.utils.data.DataLoader):
        acc = 0.0
        loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                acc += calculate_accuracy(outputs, labels)

            loss /= len(test_loader.dataset)
            acc /= len(test_loader.dataset)

            print(f"Evaluation Accuracy: {acc: .6f}")

    def save(self, model_path: str = "model.pth", is_parameter_only: bool = True):
        save_model(self.model, model_path, is_parameter_only)

    def read(self, model_path: str = "model.pth"):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def plot_result(self):
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


class CNNAutoEncoderTrainer(CNNClassificationTrainer):
    def __init__(self, model, criterion, optimizer, device=None):
        super(CNNClassificationTrainer, self).__init__(
            model, criterion, optimizer, device
        )

    def fit(
        self,
        train_loader,
        epochs,
        reshape_size=None,
        verbose_rate=1,
        validation_loader=None,
    ):
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
                loss = self.criterion(outputs, images)
                loss.backward()
                self.optimizer.step()

                # train_acc += calculate_accuracy(outputs, labels)
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
                # TODO 要変更
                # print(f"{get_epoch(epoch, epochs)} accuracy: {train_acc} loss: {train_loss}")
                pass
