import torch
import torch.nn as nn


def calculate_accuracy(true_outputs, actual_outputs, batch_size):
    tmp = true_outputs.max(1)[1] == actual_outputs
    return float(tmp.sum()) * 100.0 / batch_size


class MLPTrainer:
    def __init__(self, model, criterion, optimizer, device=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        if not device is None:
            self.device = device
        else:
            self.device = "cude" if torch.cuda.is_available() else "cpu"

    def fit(self, train_loader, epochs, reshape_size=None, verbose_rate="epoch", validation_loader=None):
        for epoch in range(epochs):
            train_loss = 0.0
            train_acc = 0.0
            val_loss = 0.0
            val_acc = 0.0

            for i, (images, labels) in enumerate(train_loader):
                if type(reshape_size) == tuple:
                    images = images.view(*reshape_size)

                images = images.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                train_acc += calculate_accuracy(outputs, labels, train_loader.batch_size)
                loss = self.criterion(outputs, labels)
                train_loss += loss.item() / train_loader.batch_size
                loss.backward()
                self.optimizer.step()

            if not validation_loader is None:
                with torch.no_grad():
                    for images, labels in validation_loader:
                        if type(reshape_size) == tuple:
                            images = images.view(*reshape_size)

                        images = images.to(self.device)
                        self.optimizer.zero_grad()
                        outputs = self.model(images)
                        val_acc += calculate_accuracy(outputs, labels, validation_loader.batch_size)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item() / validation_loader.batch_size

            print(f"Epoch [{epoch+1} / {epochs}] accuracy: {train_acc / len(train_loader)} loss: {train_loss / len(train_loader)}")

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