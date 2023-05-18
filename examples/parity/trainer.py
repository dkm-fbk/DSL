import time
import torch
from utils import *

def train(model, optimizer, loss, train_loader, test_loader, e, verbose=5, device='cpu'):
    epoch_start = time.time()

    for tX, tY in train_loader:
        tX = tX.to(device)
        tY = tY.to(device)
        optimizer.zero_grad()

        truth_values, prediction = model(tX)
        model_labels = torch.where(torch.eq(tY.view(-1, 1), prediction.view(-1, 1)), 1.0, 0.0)

        s_loss = loss(torch.logit(truth_values.view(-1, 1), 0.0001), model_labels)

        s_loss.backward()
        optimizer.step()
    accuracy_test = 0
    if e % verbose == 0 and e > 0:
        print(f'End of epoch {e}')
        print('Epoch time: ', time.time() - epoch_start)
        w, g = model.get_rules_matrix()
        print(torch.softmax(model.weight_initial, 1).cpu().detach().numpy())
        print(torch.squeeze(g))
        print(torch.squeeze(w))
        accuracy_train = test_parity(model, train_loader, device=device)
        print(f'Accuracy training: {accuracy_train}')
    accuracy_train = test_parity(model, train_loader, device=device)
    return accuracy_train

