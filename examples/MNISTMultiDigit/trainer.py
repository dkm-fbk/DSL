import time
import torch
from utils import *

def train(model, optimizer, loss, train_loader, test_loader, nn, mnist_test_data, e, run=0, device='cpu',
              verbose=20, verbose_conf=100, nn2=None):
    epoch_start = time.time()
    for i, (x, y, l) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        l = l.to(device)

        optimizer.zero_grad()

        truth_values, prediction = model(x, y)
        model_labels = torch.where(torch.eq(l, prediction), 1.0, 0.0)
        s_loss = loss(torch.logit(truth_values, 0.0001), model_labels)
        #
        s_loss.backward()
        optimizer.step()
    accuracy_train = test_sum_multi(model, train_loader, squeeze=True, device=device)

    if e % verbose == 0 and e > 0:
        print(f'End of epoch {e}')
        print('Epoch time: ', time.time() - epoch_start)


        print(f'Accuracy in sum task. Train: {accuracy_train}')

    if (e % verbose_conf == 0 and e > 0):
        confusion = test_MNIST(nn, mnist_test_data, e, 0, run, 10, device=device)



    return accuracy_train

