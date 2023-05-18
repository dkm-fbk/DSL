import time
import torch
from utils import *

def train(model, optimizer, loss, train_loader, test_loader, nn, mnist_test_data, e, run=0, device='cpu',
          verbose=50, tensorboard=False, verbose_conf=250):
    epoch_start = time.time()

    for tX, tY in train_loader:
        tX = [z.to(device) for z in tX]
        tY = tY.to(device)
        optimizer.zero_grad()

        truth_values, prediction = model(tX)
        model_labels = torch.where(torch.eq(tY.view(-1, 1), prediction.view(-1, 1)), 1.0, 0.0)

        # print(train_Y.shape)
        s_loss = loss(torch.logit(truth_values.view(-1, 1), 0.0001), model_labels)

        s_loss.backward()
        optimizer.step()

    accuracy_train = 0
    if e % verbose == 0 and e > 0:
        print(f'End of epoch {e}')
        print('Epoch time: ', time.time() - epoch_start)
        w, g = model.get_rules_matrix()
        print(torch.softmax(model.weight_initial, 1).cpu().detach().numpy())
        print(torch.squeeze(g))
        print(torch.squeeze(w))
        print('Confusion matrix')
        accuracy_train = test_visual_parity(model, train_loader, device=device)
        print(f'Accuracy training: {accuracy_train}')

    if (e % verbose_conf == 0 and e > 0):
        confusion = test_MNIST(nn, mnist_test_data, e, 0, run, 2, device=device)

    return accuracy_train

