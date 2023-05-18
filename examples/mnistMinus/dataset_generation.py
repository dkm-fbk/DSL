import torchvision
import torch
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset, DataLoader




def next_example(dataset, i, list_values, done=None, eval=False):
    x, y = next(i), next(i)
    (x, c1), (y, c2) = dataset[x], dataset[y]
    while (c1, c2) in done and not eval:
        x, y = next(i), next(i)
        (x, c1), (y, c2) = dataset[x], dataset[y]
    done.append((c1, c2))
    list_values[c1] += 1
    list_values[c2] += 1
    s__ = max(0, c1 - c2)
    label = [0.0] * 10
    label[s__] = 1.0
    label = torch.tensor([label])

    return x, y, label


def gather_examples(dataset, eval=False):
    examples = list()
    i = list(range(len(dataset)))
    random.shuffle(i)
    list_values = [0 for _ in range(10)]
    i = iter(i)
    done = []
    while(True):
        try:
            examples.append(next_example(dataset, i, list_values, done, eval))

        except StopIteration:
            break
    print(list_values)
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

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    mnist_train_data = torchvision.datasets.MNIST(root='dataset', train=True, download=True, transform=transform)
    mnist_test_data = torchvision.datasets.MNIST(root='dataset', train=False, download=True, transform=transform)
    train_data = gather_examples(mnist_train_data)
    test_data = gather_examples(mnist_test_data)
    mnist_test_data_tsne = dataloader(mnist_test_data, batch_size_val)
    train_loader = dataloader(MNISTDataset(train_data), batch_size)
    test_loader = dataloader(MNISTDataset(test_data), batch_size_val)
    return train_loader, test_loader, mnist_test_data_tsne, mnist_test_data


