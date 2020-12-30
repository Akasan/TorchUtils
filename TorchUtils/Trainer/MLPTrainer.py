import sys
import time
import warnings
from typing import List, Dict, Tuple, Union
import numpy as np

import colorama
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from colorama import Fore

from ..Core.EnvironmentChecker import convert_device, get_device_type
from ._KeyboardInterruptHandler import respond_exeption
from ._ModelSaver import save_model
from ._Printer import print_result, show_progressbar, summarize_trainer
from ._TrainerInterface import TrainerBase

warnings.simplefilter("ignore")
colorama.init()


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    return (outputs.max(1)[1] == labels).sum().item()


class MLPClassificationTrainer(TrainerBase):
    """ MLPClassificationTrainer

    Attributes:
    ----------
        model {torch.nn.Module} -- Multi Layer Perceptron's model
        criterion {torch.nn.modules.loss} -- criterion
        optimizer {torch.optim} -- optimizer
        device {str} -- device type. If you use GPU, set this as GPU (default: None)
        train_loss_history {list} -- history of training loss
        train_acc_history {list} -- history of training accuracy
        val_loss_history {list} -- history of validation loss
        val_acc_history {list} -- history of validation accuracy

    Examples:
    ---------
        >>> # this example is for MNIST classification
        >>> transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        >>> train_loader, val_loader, test_loader = load_public_dataset_with_val("MNIST", transform=transform)
        >>> model = Model(...)
        >>> criterion = nn.CrossEntropyLoss()
        >>> optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        >>> trainer = MLPClassificationTrainer(model, criterion, optimizer)
        >>> trainer.fit(train_loader, epochs=10, validation_loader=validation_loader)
        >>> trainer.plot_result()
        >>> trainer.save()
    """

    def __init__(self, model: torch.nn.Module, criterion: torch.nn.module.loss,
                 optimizer: torch.optim, device: str = None):
        """ __init__

        Arguments:
        ----------
            model {torch.nn.Module} -- Multi Layer Perceptron's model
            criterion {torch.nn.modules.loss} -- criterion
            optimizer {torch.optim} -- optimizer

        Keyword Arguments:
        ------------------
            device {str} -- device type. If you use GPU, set this as GPU (default: None)
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = get_device_type() if device is None else device
        self.train_loss_history: List[float] = []
        self.train_acc_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.val_acc_history: List[float] = []
        summarize_trainer(self.model, self.criterion, self.optimizer)

    def reset_history(self):
        """ reset_history"""
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    def fit(self, train_loader: torch.utils.data.dataloader.DataLoader, epochs: int, reshape_size: Tuple[int] = None,
            verbose_rate: int = 1, validation_loader: torch.utils.data.dataloader.DataLoader=None) -> None:
        """ train model

        Arguments:
        ----------
            train_loader {torch.utils.data.dataloader.DataLoader} -- training dataset's data loader
            epochs {int} -- the number of epochs

        Keyword Arguments:
        ------------------
            reshape_size {tuple} -- set this when you want to reshape the input data (default: None)
            verbose_rate {int} -- frequency of printing result of each epoch (default: 1)
            validation_loader {torch.utils.data.dataloader.DataLoader} -- validation dataset's data loader (default: None)

        Examples:
        ---------
            >>> mlp_trainer = MLPClassificationTrainer(model, criterion, optimizer)                     # prepare trainer
            >>> mlp_trainer.fit(train_loader, epochs=10)                                                # train model for 10 epochs
            >>> mlp_trainer.fit(train_loader, epochs=10, reshape_size=(-1, 28*29))                      # train model for 10 epochs and reshape the input (this examples is for MNIST)
            >>> mlp_trainer.fit(train_loader, epochs=10, verbose_rate=2)                                # train model for 10 epochs and print result every 2 epoch
            >>> mlp_trainer.fit(train_loader, epochs=10, validation_loader=validation_loader)           # train model for 10 epochs and use validation dataset
        """
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

                for i, (inputs, labels) in enumerate(train_loader, 1):
                    show_progressbar(len(train_loader.dataset)//train_loader.batch_size, i, whole_time=time.time()-st)

                    if type(reshape_size) == tuple:
                        inputs = inputs.view(*reshape_size)

                    inputs, labels = convert_device(inputs, labels)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
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
                        for i, (inputs, labels) in enumerate(validation_loader, 1):
                            show_progressbar(len(validation_loader.dataset)//validation_loader.batch_size, i, is_training=False)

                            if type(reshape_size) == tuple:
                                inputs = inputs.view(*reshape_size)

                            inputs, labels = convert_device(inputs, labels)
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)
                            val_acc += calculate_accuracy(outputs, labels)
                            val_loss += loss.item()

                        val_loss /= len(validation_loader.dataset)
                        val_acc /= len(validation_loader.dataset)
                        self.val_loss_history.append(val_loss)
                        self.val_acc_history.append(val_acc)

                if (epoch+1) % verbose_rate == 0:
                    print_result(epoch, epochs, train_acc, train_loss, val_acc, val_loss, time=time_diff)

        except KeyboardInterrupt:
            respond_exeption(self.model)

    def predict(self, test_loader: torch.utils.data.dataloader.DataLoader, reshape_size=None,
                to_numpy: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        """ predict

        Arguments:
        ----------
            test_loader {torch.utils.data.DataLoader} -- test dataloader

        Keyword Arguments:
        ------------------
            reshape_size {tuple} -- reshaped size (default: None)
            to_numpy {bool} -- whether predicted result will be converted to numpy or not (default: False)

        Returns:
        --------
            {torch.Tensor or np.ndarray} -- predicted result
        """
        total_outputs = None
        total_labels = None

        self.model.eval()
        for i, (inputs, labels) in enumerate(test_loader, 1):
            show_progressbar(len(test_loader.dataset)//test_loader.batch_size, i, is_training=False)

            if type(reshape_size) == tuple:
                inputss = inputss.view(*reshape_size)

            inputs = inputs.to(self.device)
            outputs = self.model(inputs)

            if total_outputs is None:
                total_outputs = outputs
                total_labels = labels
            else:
                total_outputs = torch.cat((total_outputs, outputs), 0)
                total_labels = torch.cat((total_labels, labels), 0)

        if to_numpy:
            return total_outputs.to("cpu").detach().numpy(), total_labels.to("cpu").detach().numpy()
        else:
            return total_outputs, total_labels

    def evaluate(self, test_loader: torch.utils.data.dataloader.DataLoader) -> None:
        """ evaluate

        Arguments:
        ----------
            test_loader {torch.utils.data.DataLoader} -- test loader
        """
        acc = 0.0
        loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                show_progressbar(len(test_loader.dataset)//test_loader.batch_size, i, is_training=False)
                inputs, labels = convert_device(inputs, labels)
                outputs = self.model(inputs)
                acc += calculate_accuracy(outputs, labels)
                loss += self.criterion(outputs, labels).item()

            loss /= len(test_loader.dataset)
            acc /= len(test_loader.dataset)
            print(f"Evaluation Accuracy: {acc: .6f}")

    def save(self, model_path: str = "model.pth", is_parameter_only: bool = True) -> None:
        """ save

        Keyword Arguments:
        ------------------
            model_path {str} -- file name of model (default: "model.pth")
        """
        save_model(self.model, model_path, is_parameter_only)

    def read(self, model_path: str = "model.pth") -> None:
        """ read

        Keyword Arguments:
        ------------------
            model_path {str} -- file name of saved model (default: "model.pth")
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def plot_result(self) -> None:
        """ plot_result"""
        plt.subplot(1, 2, 1)
        plt.plot(self.train_acc_history, label="train", color="r")
        if not self.val_acc_history == []:
            plt.plot(self.val_acc_history, label="val", color="b")
        plt.xlabel("epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.train_loss_history, label="train", color="r")
        if not self.val_loss_history == []:
            plt.plot(self.val_loss_history, label="val", color="b")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


# プログレスバーなど未実装
class MLPAutoEncoderTrainer(MLPClassificationTrainer):
    def __init__(self, model, criterion, optimizer, input_shape, device=None, ):
        super(MLPAutoEncoderTrainer, self).__init__(model, criterion, optimizer, device)
        self.INPUT_SHAPE = input_shape

    def fit(self, train_loader, epochs, verbose_rate=1, validation_loader=None):
        try:
            for epoch in range(epochs):
                train_loss = 0.0
                correct_num = 0.0

                self.model.train()

                for images, labels in train_loader:
                    images = images.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, images)
                    loss.backward()
                    self.optimizer.step()

                    correct_num += calculate_accuracy(outputs, labels)
                    train_loss += loss.item()

                train_loss /= len(train_loader.dataset)
                train_acc = correct_num / len(train_loader.dataset)
                self.train_loss_history.append(train_loss)
                self.train_acc_history.append(train_acc)

                if not validation_loader is None:
                    val_loss = 0.0
                    val_acc = 0.0

                    self.model.eval()
                    with torch.no_grad():
                        for images, labels in validation_loader:
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
                    print_result(epoch, epochs, train_acc, train_loss, val_acc, val_loss)

        except KeyboardInterrupt:
            respond_exeption(self.model)

    def predict(self, test_loader, reshape_size=None, to_numpy=False):
        total_outputs = None
        total_labels = None

        self.model.eval()

        for images, labels in test_loader:
            if type(reshape_size) == tuple:
                images = images.view(*reshape_size)

            images = images.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

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
