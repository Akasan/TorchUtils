import sys
sys.path.append("../")
from TorchUtils.ModelGenerator.MLP import MLP
from TorchUtils.DatasetGenerator import FromFolder
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim

EPOCH = 1
BATCH_SIZE = 2


if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((480, 640)), transforms.ToTensor()])
    train_loader = FromFolder.generate_dataset_loader("data/train", transform, batch_size=BATCH_SIZE)
    test_loader = FromFolder.generate_dataset_loader("data/test", transform)

    neurons = [640*480*3, 100, 24]
    activations = ["relu", "softmax"]
    model = MLP(neurons, activations)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 640*480*3)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(loss)