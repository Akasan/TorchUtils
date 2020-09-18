import torch
import warnings
warnings.simplefilter("ignore")


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.data = data.detach().numpy()
        self.labels = labels.detach().numpy()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]

        if self.transform is not None:
            data = self.transform(data)

        return data, label

    def size(self):
        return self.data.size()