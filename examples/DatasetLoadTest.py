import sys
sys.path.append("../")
from TorchUtils.DatasetGenerator import FromPublicDatasets

train_loader, test_loader = FromPublicDatasets.load_public_datasets("FashionMNIST")
print(type(train_loader))
