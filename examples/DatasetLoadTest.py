import sys
sys.path.append("../")
from TorchUtils.DatasetGenerator import FromPublicDatasets

train_loader, val_loader, test_loader = FromPublicDatasets.load_public_dataset_with_val("FashionMNIST")
print(train_loader, val_loader, test_loader)
