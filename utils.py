import numpy as np
import torch
from torch.autograd import Variable
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors


def F1_compute(N, max_digit, confusion):
    print(confusion)
    F1 = 0
    for nr in range(max_digit):
        TP = confusion[nr, nr]
        FP = sum(confusion[:, nr]) - TP
        FN = sum(confusion[nr, :]) - TP
        F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N
    print('F1: ', F1)
    return


def accuracy_rules(weights, p):
    rules_matrix_final = np.zeros([10, 10])

    for i in range(10):
        for j in range(10):
            x = p[i]
            y = p[j]
            #print(f'{i} + {j} = {r[x, y]}')
            rules_matrix_final[i, j] = torch.argmax(weights[x, y])
    return

def swap_conf(confusion):
    _, p = torch.max(torch.tensor(confusion.astype(np.float32)), 1)
    return torch.tensor(confusion.astype(np.int32))[:, p].cpu().numpy(), p

def swap_rules(rules, p):
    return torch.tensor(rules)[p, :][:, p]

def visualize_confusion(confusion, name, indices="0123456789"):
    df_cm = pd.DataFrame(confusion, index=[i for i in indices],
                         columns=[i for i in indices])
    plt.figure(figsize=(10, 7))
    ax = sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='d', cbar=False, annot_kws={'size':15})
    ax.set_xticks([i for i in range(10)])
    ax.set_yticks([i for i in range(10)])
    ax.set_yticklabels([i for i in range(10)], rotation=90)
    ax.set_xticklabels([i for i in range(10)], rotation=0)

    ax.tick_params(axis='both', which='both', length=3, labelsize=15)
    plt.savefig('./visualizations/{}_confusion.png'.format(name))
    plt.close()

    return


def test_MNIST(model, dataset, epoch, folder, num_exp, n_digits, device='cpu'):
    confusion = np.zeros((n_digits, n_digits), dtype=np.uint32)  # First index actual, second index predicted
    N = 0
    for d, l in dataset:
        if l < n_digits:
            N += 1
            d = Variable(d.unsqueeze(0))
            d = d.to(device)
            with torch.no_grad():
                outputs, _ = model(d)
                _, out = torch.max(outputs.data, 1)
                out = out.to(device)
                c = int(out.squeeze())
            confusion[l, c] += 1
    print()
    print(confusion)
    return confusion #, p

def test_MNIST_visual(model, dataset, n_digits, device='cpu'):
    confusion = np.zeros((n_digits, n_digits), dtype=np.uint32)  # First index actual, second index predicted
    N = 0
    for d, l in dataset:
        if l < n_digits:
            N += 1
            d = Variable(d.unsqueeze(0))
            d = d.to(device)
            with torch.no_grad():
                outputs, _ = model(d)
                _, out = torch.max(outputs.data, 1)
                out = out.to(device)
                c = int(out.squeeze())
            confusion[l, c] += 1
    print()
    print(confusion)
    return confusion #, p

def test_EMNIST(model, emnist_test_data, epoch, folder, num_exp, max_digit=4, device='cpu'):
    confusion = np.zeros((max_digit, max_digit), dtype=np.uint32)  # First index actual, second index predicted
    N = 0
    for d, l in emnist_test_data:
        if l < max_digit:
            N += 1
            d = Variable(d.unsqueeze(0))
            d = d.to(device)
            with torch.no_grad():
                outputs, _ = model(d)
                _, out = torch.max(outputs.data, 1)
                c = int(out.squeeze())
            confusion[l, c] += 1
    print(confusion)
    return confusion,



def test_sum(model, dataloader, device='cpu'):
    x, y, l = next(iter(dataloader))
    x = x.to(device)
    y = y.to(device)
    l = l.to(device)

    with torch.no_grad():
        _, prediction = model(x, y, eval=True)
        prediction.to(device)
        _, label = torch.max(torch.squeeze(l), 1)

    return torch.sum(label.to(device) == torch.squeeze(prediction)).float() / label.shape[0]


def test_sum_multi(model, dataloader, squeeze=True, device='cpu'):
    x, y, l = next(iter(dataloader))
    x = x.to(device)
    y = y.to(device)
    l = l.to(device)

    with torch.no_grad():
        _, prediction = model(x, y, eval=True)
        prediction.to(device)

    if squeeze:
        return torch.sum(torch.all(l.to(device) == torch.squeeze(prediction), dim=1)).float() / l.shape[0]
    else:
        return torch.sum(torch.all(l.to(device) == prediction, dim=1)).float() / l.shape[0]

def test_sum_multi_single(model, dataloader, device='cpu'):
    x, y, l = next(iter(dataloader))
    x = x.to(device)
    y = y.to(device)
    l = l.to(device)

    with torch.no_grad():
        _, prediction = model(x, y, eval=True)
        prediction.to(device)

    return torch.sum(torch.sum(l.to(device) == torch.squeeze(prediction), dim=1)/l.shape[1]).float() / l.shape[0]



def test_parity(model, dataloader, device='cpu'):
    x, y = next(iter(dataloader))
    x = x.to(device)
    y = y.to(device)
    _, predictions = model(x, eval=True)

    return torch.sum(predictions.view(-1, 1) == y) / x.shape[0]


def test_visual_parity(model, dataloader, device='cpu'):
    x, y = next(iter(dataloader))
    x = [z.to(device) for z in x]
    y = y.to(device)

    _, predictions = model(x, eval=True)

    return torch.sum(predictions.view(-1) == y.to(device)) / y.shape[0]


def test_multiop(model, dataloader, device='cpu'):
    x, y, a, l = next(iter(dataloader))
    x = x.to(device)
    y = y.to(device)
    a = a.to(device)
    l = l.to(device)

    with torch.no_grad():
        _, prediction = model(x, y, a, eval=True)
        prediction.to(device)
        _, label = torch.max(torch.squeeze(l), 1)
    return torch.sum(label == prediction.view(-1)) / label.shape[0]


def visualize_rules(rules, name):
    fig, ax = plt.subplots(figsize=(7, 7))
    rules=rules.cpu().numpy().squeeze()
    cmap = colors.ListedColormap(['#7b68ee'])

    ax.set_xticks([i for i in range(10)])
    ax.set_yticks([i for i in range(10)])
    ax.set_yticklabels([i for i in range(10)], rotation=90)
    ax.invert_xaxis()
    ax.xaxis.tick_top()

    ax.tick_params(axis='both', which='both', length=0, labelsize=15)
    ax.imshow(rules, vmin=0, vmax=0, cmap=cmap)
    for i in range(10):
        for j in range(10):
            ax.text(i, j, str(rules[i][j]), ha="center", va="center", color="white", fontsize=15)

    plt.savefig('./visualizations/{}_rules.png'.format(name))
    plt.close()