import time
import torch
from utils import *
import random
import numpy as np
import os


def train(model, optimizer, loss, train_loader, test_loader, nn, mnist_test_data, e, run=0, device='cpu',
          verbose=20, verbose_conf=10, nn2=None):
    epoch_start = time.time()
    for i, (x, y, l) in enumerate(train_loader):

        x = x.to(device)
        y = y.to(device)
        l = l.to(device)

        _, labels = torch.max(torch.squeeze(l), 1)
        optimizer.zero_grad()

        truth_values, prediction = model(x, y)
        model_labels = torch.where(torch.eq(labels, prediction), 1.0, 0.0).view(-1)

        s_loss = loss(torch.logit(truth_values, 0.0001), model_labels)

        s_loss.backward()
        optimizer.step()
    accuracy_train = test_sum(model, train_loader, device=device)
    if e % verbose == 0 and e > 0:
        print(f'End of epoch {e}')
        print('Epoch time: ', time.time() - epoch_start)


        print(f'Accuracy in sum task. Train: {accuracy_train}')

    if (e % verbose_conf == 0 and e > 0):
        confusion = test_MNIST(nn, mnist_test_data, e, 0, run, n_digits=10, device=device)
        if nn2 is not None:
            confusion2 = test_MNIST(nn2, mnist_test_data, e, 0, run, n_digits=10, device=device)
    return accuracy_train

