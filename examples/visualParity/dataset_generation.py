
import torchvision
import torch
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

import os
import torch
from torch.utils.data import Dataset, DataLoader
import random


def symbolic_parity_generation(n_samples, length):
    parity_dataset = []
    for i in range(1000):
        x = [random.randint(0, 1) for _ in range(length)]
        y = sum(x) % 2
        parity_dataset.append((x,y))

    return parity_dataset




def select_mnist_samples(mnist, j):
    zeros = []
    ones = []

    for k in j:
        x, y = mnist[k]
        if y == 0:
            zeros.append(x)
        if y == 1:
            ones.append(x)

    return zeros, ones


def next_example(parity, k, zeros, ones):
    i = next(k)
    bit_sequence, label = parity[i]

    images_list = []
    for digit in bit_sequence:
        if digit == 0:
            images_list.append(random.choice(zeros))
        elif digit == 1:
            images_list.append(random.choice(ones))

    return images_list, label


def gather_examples(mnist, parity):
    j = list(range(len(mnist)))
    j = iter(j)
    zeros, ones = select_mnist_samples(mnist, j)

    k = list(range(len(parity)))
    random.shuffle(k)
    k = iter(k)

    examples = list()
    while (True):
        try:
            examples.append(next_example(parity, k, zeros, ones))
        except StopIteration:
            break
    return examples


class ParityDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def dataloader(dataset, batch_size=32):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )





def get_dataset(batch_size, batch_size_val, eval=False):
    if not eval:
        random.seed(0)
    train_parity = symbolic_parity_generation(900, 4)
    test_parity = symbolic_parity_generation(100, 100)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])  # , 0.5), (0.5, 0.5, 0.5))])
    mnist_train_data = torchvision.datasets.MNIST(root='dataset', train=True, download=True, transform=transform)
    mnist_test_data = torchvision.datasets.MNIST(root='dataset', train=False, download=True, transform=transform)

    train_data = gather_examples(mnist_train_data, train_parity)
    test_data = gather_examples(mnist_test_data, test_parity)

    train_loader = dataloader(ParityDataset(train_data), batch_size)
    test_loader = dataloader(ParityDataset(test_data), len(test_data))
    return train_loader, test_loader, mnist_train_data, mnist_test_data


