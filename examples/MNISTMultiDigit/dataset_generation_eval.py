import torchvision
import torch
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset, DataLoader


def next_example(dataset, i, num_digits, zeros_images, padding=True):
    number_x = ''
    number_y = ''

    x = []
    y = []
    if padding:
        x.append(random.choice(zeros_images))
        y.append(random.choice(zeros_images))

    for _ in range(num_digits):
        x1 = next(i)
        (x1, c1) = dataset[x1]
        x.append(x1)
        number_x += str(c1)

    for _ in range(num_digits):  # TODO: rimuovere codice ripetuto
        y1 = next(i)
        (y1, c2) = dataset[y1]
        y.append(y1)
        number_y += str(c2)

    label = int(number_x) + int(number_y)
    label = list(str(label))
    label = [int(digit) for digit in label]

    output_length = num_digits + padding

    while len(label) < output_length:
        label = [0]+label
    while len(label) > output_length:
        label = label[1:]

    return torch.concat(x), torch.concat(y), torch.Tensor(label)



#
# def next_example(dataset, i, num_digits, done, zeros, num_zeros):
#     number_x = ''
#     number_y = ''
#
#     x = []
#     number_x = '0'*num_zeros
#     for _ in range(num_zeros):
#         x.append(random.choice(zeros))
#     for _ in range(num_digits):
#         x1 = next(i)
#         (x1, c1) = dataset[x1]
#         x.append(x1)
#         number_x += str(c1)
#     y = []
#     number_y = '0'*num_zeros
#     for _ in range(num_zeros):
#         y.append(random.choice(zeros))
#     for _ in range(num_digits):
#         y1 = next(i)
#         (y1, c2) = dataset[y1]
#         y.append(y1)
#         number_y += str(c2)
#     label = int(number_x) + int(number_y)
#     label = list(str(label))
#     label = [int(s) for s in label]
#     while len(label) <= num_zeros + 1:
#         label = [0]+label
#     label = torch.Tensor(label[1:])  # TODO: rimuovere [1:] e + 1 da riga 37
#
#     return torch.concat(x), torch.concat(y), label


def gather_examples(dataset, num_digits, padding=True, mode=0):
    examples = list()
    done = []
    iters=1
    zeros_images = []
    i = list(range(len(dataset)))
    random.shuffle(i)
    i = iter(i)
    while (True):
        try:
            tmp = next(i)
            x, c = dataset[tmp]
            if c == 0:
                zeros_images.append(x)
        except StopIteration:
            break
    for jj in range(iters):
        i = list(range(len(dataset)))
        random.shuffle(i)
        i = iter(i)
        while(True):
            try:
                examples.append(next_example(dataset, i, num_digits, zeros_images, padding))
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




def get_dataset_eval(batch_size, batch_size_val, N):


    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])  # , 0.5), (0.5, 0.5, 0.5))])
    mnist_train_data = torchvision.datasets.MNIST(root='dataset', train=True, download=True, transform=transform)
    mnist_test_data = torchvision.datasets.MNIST(root='dataset', train=False, download=True, transform=transform)

    train_data_complex = gather_examples(mnist_train_data, N, True, 1)
    test_data_complex = gather_examples(mnist_test_data, N, True, 1)

    # mnist_test_data_tsne = dataloader(mnist_test_data, s.batch_size_val)

    train_loader_complex = dataloader(MNISTDataset(train_data_complex), batch_size)
    test_loader_complex = dataloader(MNISTDataset(test_data_complex), batch_size_val)
    return train_loader_complex, test_loader_complex, mnist_train_data, mnist_test_data


