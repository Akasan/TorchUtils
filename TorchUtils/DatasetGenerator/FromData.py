import torch
from ._LoaderGenerator import generate_dataloader


class UserDataDataset(torch.utils.data.Dataset):
    """ユーザが指定したデータからDatasetを作成する

    例えば一般的な数値ベクトルなどを指定すると、それに対応するDatasetを自動で生成します。
    データセットのサイズは入力Xのシェイプの最初の要素となります。
    """

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def generate_dataset_for_userdata(
    X, y, batch_size, shuffle=True, num_workers=2, transform=None
):
    dataset = UserDataDataset(X, y, transform)
    return generate_dataloader(dataset, batch_size, shuffle, num_workers)
