import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.view(-1, 12 * 12 * 64)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


model = MyModel()
pickle.dump(model, open("hoge.mdl", "wb"))
