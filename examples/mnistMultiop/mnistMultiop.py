import os
from tqdm import tqdm
from utils import *
import madgrad
from examples.mnistMultiop.dataset_generation import *
from examples.mnistMultiop.models import *
from examples.mnistMultiop.trainer import *
import torch
import random
import numpy as np
import optuna
import pickle
import argparse
#[I 2023-01-18 16:07:13,860] Trial 22 finished with value: 0.296875 and parameters: {'EPSILON_SYMBOLS': 0.23699542018849643, 'EPSILON_LETTERS': 0.647002418144579, 'EPSILON_RULES': 0.2634794214513527, 'LR': 0.14814748838947028}. Best is trial 22 with value: 0.296875.

def experiment_eval(args):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    EXPERIMENT = 'mnistMultiop'
    NUM_EXPERIMENT = 1
    DEVICE = 'cuda:0'
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 1000
    TSNE = False
    TSNE_EPOCH = 1000
    CKPT_SAVE = 250
    EPOCHS = 20000
    EPSILON_SYMBOLS = args.eps_sym
    EPSILON_LETTERS = args.eps_let
    EPSILON_RULES = args.eps_rul
    LR = args.lr



    train_loader, test_loader, mnist_test_data_tsne, emnist_test_data_tsne, mnist_test_data, emnist_test_data = \
        get_dataset(BATCH_SIZE, BATCH_SIZE_VAL)
    accuracy_train_results = []



    nn = MNIST_Net().to(DEVICE)
    nn_letters = MNIST_Net(N=4).to(DEVICE)
    model = MNISTModelMultiop(nn, nn_letters, EPSILON_SYMBOLS, EPSILON_LETTERS, EPSILON_RULES, device=DEVICE).to(DEVICE)
    optimizer = madgrad.MADGRAD(
        [{'params': list(model.parameters())[:1]}, {'params': list(model.parameters())[1:], 'lr': 1e-3}], lr=LR)
    loss = torch.nn.BCEWithLogitsLoss()
    print('Accuracy in multiop task (before training):' + str(test_multiop(model, test_loader, DEVICE)))
    # Learning
    for e in tqdm(range(EPOCHS)):
        accuracy = train(model, optimizer, loss, train_loader, test_loader, nn, nn_letters, mnist_test_data,
                         emnist_test_data, e, run=0, device=DEVICE)

        if CKPT_SAVE > 0 and e > 0 and e % CKPT_SAVE == 0:
            if not os.path.exists('./experiments/'):
                os.mkdir('./experiments/')
            if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
            torch.save(model.state_dict(),
                       './experiments/ckpt_{}/ckpt.{}.pth'.format(EXPERIMENT, e))
        if accuracy > 0.975:
            if not os.path.exists('./experiments/'):
                os.mkdir('./experiments/')
            if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
            torch.save(model.state_dict(),
                       './experiments/ckpt_{}/ckpt_final.pth'.format(EXPERIMENT, e))
            break
    accuracy_test = []
    for e in range(10):
        accuracy = test_multiop(model, test_loader, device=DEVICE)
        accuracy_test.append(accuracy.cpu())
    acc_test_np = np.array(accuracy_test)

    print('Experiment is over. After {} runs on training set we obtained: \n Mean: {}\n Std: {}'.format(NUM_EXPERIMENT,
                                                                                                        np.mean(
                                                                                                            acc_test_np),
                                                                                                        np.std(
                                                                                                            acc_test_np)))
    return np.mean(acc_test_np)


def experiment(trial=None):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    EXPERIMENT = 'mnistMultiop'
    NUM_EXPERIMENT = 1
    DEVICE = 'cuda:0'
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 1000
    TSNE = False
    TSNE_EPOCH = 1000
    CKPT_SAVE = 50
    EPOCHS = 300
    EPSILON_SYMBOLS = trial.suggest_float('EPSILON_SYMBOLS', 0.0, 0.8)
    EPSILON_LETTERS = trial.suggest_float('EPSILON_LETTERS', 0.0, 0.8)
    EPSILON_RULES = trial.suggest_float('EPSILON_RULES', 0.0, 0.8)
    LR = trial.suggest_float('LR', 5e-4, 5e-1, log=True)



    train_loader, test_loader, mnist_test_data_tsne, emnist_test_data_tsne, mnist_test_data, emnist_test_data = \
        get_dataset(BATCH_SIZE, BATCH_SIZE_VAL)
    accuracy_train_results = []



    nn = MNIST_Net().to(DEVICE)
    nn_letters = MNIST_Net(N=4).to(DEVICE)
    model = MNISTModelMultiop(nn, nn_letters, EPSILON_SYMBOLS, EPSILON_LETTERS, EPSILON_RULES, device=DEVICE).to(DEVICE)

    optimizer = madgrad.MADGRAD(
        [{'params': list(model.parameters())[:1]}, {'params': list(model.parameters())[1:], 'lr': 1e-3}], lr=LR)
    loss = torch.nn.BCEWithLogitsLoss()
    print('Accuracy in multiop task (before training):' + str(test_multiop(model, test_loader, DEVICE)))
    # Learning
    for e in tqdm(range(EPOCHS)):
        accuracy = train(model, optimizer, loss, train_loader, test_loader, nn, nn_letters, mnist_test_data,
                         emnist_test_data, e, run=0, device=DEVICE)

        if CKPT_SAVE > 0 and e > 0 and e % CKPT_SAVE == 0:
            if not os.path.exists('./experiments/'):
                os.mkdir('./experiments/')
            if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
            torch.save(model.state_dict(),
                       './experiments/ckpt_{}/ckpt.{}.pth'.format(EXPERIMENT, e))
        if accuracy > 0.97:
            if not os.path.exists('./experiments/'):
                os.mkdir('./experiments/')
            if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
            torch.save(model.state_dict(),
                       './experiments/ckpt_{}/ckpt_final.pth'.format(EXPERIMENT, e))
            break
        trial.report(accuracy, e)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    acc_train_np = np.array([accuracy.cpu().numpy()])

    print('Experiment is over. After {} runs on training set we obtained: \n Mean: {}\n Std: {}'.format(NUM_EXPERIMENT,
                                                                                                        np.mean(
                                                                                                            acc_train_np),
                                                                                                        np.std(
                                                                                                            acc_train_np)))
    return np.mean(acc_train_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval',
                        action='store_true', default=False)
    parser.add_argument('-c', '--checkpoint', default='')
    parser.add_argument('-o', '--optuna', action='store_true', default=False)
    parser.add_argument('-lr', '--lr', default=0.14814748838947028)
    parser.add_argument('-es', '--eps_sym', default=0.23699542018849643)
    parser.add_argument('-el', '--eps_let', default=0.647002418144579)
    parser.add_argument('-er', '--eps_rul', default=0.2634794214513527)
    args = parser.parse_args()
    if args.optuna:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
        study.optimize(experiment, n_trials=100)
        pickle.dump(study, open('./studies/mnist_optuna.pkl', 'wb'))
    elif not args.optuna:
        experiment_eval(args)
    else:
        print('There is an error in your configuration!. ')



