import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ._Base import TrainerBase
from ..Core.EnvironmentChecker import get_device_type
from ._Printer import get_result_text


def calculate_accuracy(outputs, labels):
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

    def __init__(self, model, criterion, optimizer, device=None):
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
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    def reset_history(self):
        """ reset_history"""
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    def fit(self, train_loader, epochs, reshape_size=None, verbose_rate=1, validation_loader=None):
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
                        loss = self.criterion(outputs, labels)
                        val_acc += calculate_accuracy(outputs, labels)
                        val_loss += loss.item()

                    val_loss /= len(validation_loader.dataset)
                    val_acc /= len(validation_loader.dataset)
                    self.val_loss_history.append(val_loss)
                    self.val_acc_history.append(val_acc)

            if (epoch+1) % verbose_rate == 0:
                print(get_result_text(epoch, epochs, train_acc, train_loss, val_acc, val_loss))

    def predict(self, test_loader, reshape_size=None):
        for images, labels in test_loader:
            if type(reshape_size) == tuple:
                images = images.view(*reshape_size)

            images = images.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

    def save(self, model_path="model.pth"):
        """ save

        Keyword Arguments:
        ------------------
            model_path {str} -- file name of model (default: "model.pth")
        """
        torch.save(self.model.state_dict(), model_path)

    def read(self, model_path="model.pth"):
        """ read

        Keyword Arguments:
        ------------------
            model_path {str} -- file name of saved model (default: "model.pth")
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def plot_result(self):
        """ plot_result"""
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
        self.device = get_device_type() if device is None else device
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    def fit(self, train_loader, epochs, reshape_size=None, verbose_rate=1, validation_loader=None):
        for epoch in range(epochs):
            train_loss = 0.0
            correct_num = 0.0

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
                print(get_result_text(epoch, epochs, train_acc, train_loss, val_acc, val_loss))

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