import torchvision
import torch
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



def next_example(dataset, i, ds_letters, j):
    x, y = next(i), next(i)
    (x, c1), (y, c2) = dataset[x], dataset[y]
    a = next(j)
    a, cj = ds_letters[a]
    while (cj == 1 and c1 < c2) or (cj == 2 and (c2 == 0 or c1 < c2)):
        a = next(j)
        a, cj = ds_letters[a]
    if cj == 0:
       s = c1 + c2
    elif cj == 1:
       s = c1 - c2
    elif cj == 3:
       s = (c1 * c2)
    elif cj == 2:
       s = c1 // c2
    else:
        print(cj, 'Error')
    label = [0.0] * 82
    label[s] = 1.0
    label = torch.tensor([label])

    return x, y, a, label



def gather_examples(dataset, letters, filename, mode=0):
    ds_letters = list()
    cl = [0, 0, 0, 0]
    for x, c in letters:
        if c < 5:
           cl[c-1] += 1
           ds_letters.append((x, c-1))

    print(cl)
    examples = list()
    if mode == 0:
        iters = 1
    else:
        iters = 1
    for jj in tqdm(range(iters)):
        i = list(range(len(dataset)))
        random.shuffle(i)
        i = iter(i)
        j = list(range(len(ds_letters)))
        random.shuffle(j)
        j = iter(j)
        while(True):
            try:
                examples.append(next_example(dataset, i, ds_letters, j))
            except StopIteration:
                break


    return examples


class MNISTDataset(Dataset):
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
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])  # , 0.5), (0.5, 0.5, 0.5))])
    transform2 = torchvision.transforms.Compose([
        lambda img: torchvision.transforms.functional.rotate(img, -90),
        lambda img: torchvision.transforms.functional.hflip(img),
        torchvision.transforms.ToTensor()
    ])

    mnist_train_data = torchvision.datasets.MNIST(root='dataset', train=True, download=True, transform=transform)
    mnist_test_data = torchvision.datasets.MNIST(root='dataset', train=False, download=True, transform=transform)
    emnist_letters_train = torchvision.datasets.EMNIST(root='./emnist_data/', split='letters', train=True,
                                                       transform=transform2, download=True)
    emnist_letters_val = torchvision.datasets.EMNIST(root='./emnist_data/', split='letters', train=True,
                                                     transform=transform2, download=True)

    train_data = gather_examples(mnist_train_data, emnist_letters_train, 'train_data_multiop')
    test_data = gather_examples(mnist_test_data, emnist_letters_val, 'test_data_multiop', 1)

    emnist_test_data = list()
    for x, c in emnist_letters_val:
        if c < 5:
            emnist_test_data.append((x, c - 1))
    mnist_test_data_tsne = dataloader(mnist_test_data, batch_size_val)
    emnist_test_data_tsne = dataloader(emnist_test_data, batch_size_val)

    train_loader = dataloader(MNISTDataset(train_data), batch_size)
    test_loader = dataloader(MNISTDataset(test_data), batch_size_val)
    return train_loader, test_loader, mnist_test_data_tsne, emnist_test_data_tsne, mnist_test_data, emnist_test_data


