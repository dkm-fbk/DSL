import os
from tqdm import tqdm
from utils import *
import madgrad
from examples.visualParity.dataset_generation import *
from examples.visualParity.models import *
from examples.visualParity.trainer import *
import torch
import random
import numpy as np
import optuna
import pickle

import argparse


def experiment_eval(args):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    NUM_EXPERIMENT = 1
    DEVICE = 'cpu'
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 1000
    CKPT_SAVE = 50
    EPOCHS = 500
    EPSILON_SYMBOLS = args.eps_sym
    EPSILON_RULES = args.eps_rul
    LR = args.lr

    EXPERIMENT = 'visualParity'

    train_loader, test_loader, mnist_train_data, mnist_test_data = get_dataset(BATCH_SIZE, BATCH_SIZE_VAL)
    accuracy_train_results = []

    print('Starting training number ', 0)


    nn = MNIST_Net(N=2).to(DEVICE)
    model = VisualParityFunctionModel(nn, EPSILON_SYMBOLS, EPSILON_RULES, device=DEVICE).to(DEVICE)
    optimizer = madgrad.MADGRAD(
        [{'params': list(model.parameters())[:1]}, {'params': list(model.parameters())[1:], 'lr': 1e-3}], lr=LR)
    loss = torch.nn.BCEWithLogitsLoss()
    print('Accuracy in parity task (before training):' + str(test_visual_parity(model, train_loader, device=DEVICE)))
    # Learning
    for e in tqdm(range(EPOCHS)):
        accuracy = train(model, optimizer, loss, train_loader, test_loader, nn, mnist_test_data, e,
                             run=0, device=DEVICE)

        if CKPT_SAVE > 0 and e > 0 and e % CKPT_SAVE == 0:
            if not os.path.exists('./experiments/'):
                os.mkdir('./experiments/')
            if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
            torch.save(model.state_dict(),
                       './experiments/ckpt_{}/ckpt.{}.pth'.format(EXPERIMENT, e))

    accuracy_test_results = []
    for e in range(10):
        accuracy_test = test_visual_parity(model, test_loader, device=DEVICE)
        accuracy_test_results.append(accuracy_test.cpu().numpy())
    accuracy_test_results.append(accuracy_test.cpu().numpy())

    acc_test_np = np.array(accuracy_test_results)

    print('Experiment is over. After {} runs on training set we obtained: \n Mean: {}\n Std: {}'.format(NUM_EXPERIMENT,
                                                                                                        np.mean(
                                                                                                            acc_test_np),
                                                                                                        np.std(
                                                                                                            acc_test_np)))

    return np.mean(acc_test_np)


def experiment(trial):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    NUM_EXPERIMENT = 1
    DEVICE = 'cuda:0'
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 1000
    CKPT_SAVE = 50
    EPOCHS = 500
    EPSILON_SYMBOLS = trial.suggest_float('EPSILON_SYMBOLS', 0.0, 0.8)
    EPSILON_RULES = trial.suggest_float('EPSILON_RULES', 0.0, 0.8)
    LR = trial.suggest_float('LR', 5e-4, 5e-1, log=True)

    EXPERIMENT = 'visualParity'

    train_loader, test_loader, mnist_train_data, mnist_test_data = get_dataset(BATCH_SIZE, BATCH_SIZE_VAL)
    accuracy_train_results = []

    print('Starting training number ', 0)


    nn = MNIST_Net(N=2).to(DEVICE)
    model = VisualParityFunctionModel(nn, EPSILON_SYMBOLS, EPSILON_RULES, device=DEVICE).to(DEVICE)
    optimizer = madgrad.MADGRAD(
        [{'params': list(model.parameters())[:1]}, {'params': list(model.parameters())[1:], 'lr': 1e-3}], lr=LR)
    loss = torch.nn.BCEWithLogitsLoss()
    print('Accuracy in parity task (before training):' + str(test_visual_parity(model, train_loader, device=DEVICE)))
    # Learning
    for e in tqdm(range(EPOCHS)):
        accuracy = train(model, optimizer, loss, train_loader, test_loader, nn, mnist_test_data, e,
                             run=0, device=DEVICE)

        if CKPT_SAVE > 0 and e > 0 and e % CKPT_SAVE == 0:
            if not os.path.exists('./experiments/'):
                os.mkdir('./experiments/')
            if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
            torch.save(model.state_dict(),
                       './experiments/ckpt_{}/ckpt.{}.pth'.format(EXPERIMENT, e))

    accuracy_train = test_visual_parity(model, train_loader, device=DEVICE)
    accuracy_train_results.append(accuracy_train.cpu().numpy())

    acc_train_np = np.array(accuracy_train_results)

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
    parser.add_argument('-lr', '--lr', default=0.13074863877410006)
    parser.add_argument('-es', '--eps_sym', default=0.10836478372618243)
    parser.add_argument('-er', '--eps_rul', default=0.6332228380367143)
    args = parser.parse_args()
    if args.optuna:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
        study.optimize(experiment, n_trials=100)
        pickle.dump(study, open('./studies/visual_parity_optuna.pkl', 'wb'))
    elif not args.optuna:
        experiment_eval(args)
    else:
        print('There is an error in your configuration!. ')


