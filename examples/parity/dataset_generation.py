import os
import torch
from torch.utils.data import Dataset, DataLoader
import random




class BoolsDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, idx):
        tx, ty = self.dataset
        return tx[idx], ty[idx]


def dataloader(dataset, batch_size=32):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )


def get_dataset(batch_size, batch_size_val=1000, parity_sequence_length=20, eval=False):
    if not eval:
        random.seed(0)
    if not os.path.exists('dataset/parity'):
        os.system('wget -cq powei.tw/parity.zip -P ./dataset && unzip -d ./dataset -qq ./dataset/parity.zip')
    with open(f'./dataset/parity/{parity_sequence_length}/features.pt', 'rb') as f:
        X = torch.load(f)
    with open(f'./dataset/parity/{parity_sequence_length}/labels.pt', 'rb') as f:
        Y = torch.load(f)
    train_X = X[:9000, :]
    train_Y = Y[:9000, :]

    test_X = X[9000:, :]
    test_Y = Y[9000:, :]

    train_loader = dataloader(BoolsDataset((train_X, train_Y)), batch_size)
    test_loader = dataloader(BoolsDataset((test_X, test_Y)), batch_size_val)
    return train_loader, test_loader


