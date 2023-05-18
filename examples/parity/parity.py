import os
from tqdm import tqdm
from utils import *
import madgrad
from examples.parity.dataset_generation import *
from examples.parity.models import *
from examples.parity.trainer import *
import torch
import random
import numpy as np
import optuna
import pickle
import argparse

def eval(LOAD_CKPT):
    NUM_VAL_ITERS = 10
    DEVICE = 'cuda:0'
    BATCH_SIZE = 1024
    BATCH_SIZE_VAL = 1000
    nn = MNIST_Net().to(DEVICE)
    model = MNISTMinusModel(nn, 0.2, 0.2, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(LOAD_CKPT))
    model.eval()
    accuracy_test_results = []
    for e in range(NUM_VAL_ITERS):
        train_loader, test_loader, mnist_test_data_tsne, mnist_test_data = get_dataset(BATCH_SIZE, BATCH_SIZE_VAL, 100, eval=True)

        accuracy_test = test_parity(model, test_loader, device=DEVICE)
        accuracy_test_results.append(accuracy_test.cpu().numpy())


    acc_test = np.array(accuracy_test_results)
    print('Experiment is over. After {} runs on test set we obtained: \n Mean: {}\n Std: {}'.format(NUM_VAL_ITERS,
                                                                                                    np.mean(
                                                                                                        acc_test),
                                                                                                    np.std(
                                                                                                        acc_test)))
    return


def experiment(optimize_hp, trial=None, args=None):
    EXPERIMENT = 'parity'
    NUM_EXPERIMENT = 1
    DEVICE = 'cuda:0'
    BATCH_SIZE = 1024
    BATCH_SIZE_VAL = 1000
    TSNE = False
    TSNE_EPOCH = 1000
    CKPT_SAVE = 50
    EPOCHS = 50
    if optimize_hp:
        EPSILON_RULES = trial.suggest_float('EPSILON_RULES', 0.0, 0.8)
        LR = trial.suggest_float('LR', 5e-4, 5e-1, log=True)
    else:
        EPSILON_RULES = args.eps_rul
        LR = args.lr


    train_loader, test_loader = get_dataset(BATCH_SIZE, BATCH_SIZE_VAL, 20)
    accuracy_train_results = []
    accuracy_test_results = []

    for num_exp in range(NUM_EXPERIMENT):
        print('Starting training number ', num_exp)
        torch.manual_seed(num_exp)
        random.seed(num_exp)
        np.random.seed(num_exp)
        model = ParityFunctionModel(EPSILON_RULES, DEVICE).to(DEVICE)

        optimizer = madgrad.MADGRAD(
            [{'params': list(model.parameters())[:1]}, {'params': list(model.parameters())[1:], 'lr': 1e-3}], lr=LR)
        loss = torch.nn.BCEWithLogitsLoss()
        print('Accuracy in parity task (before training):' + str(test_parity(model, train_loader, DEVICE)))
        # Learning
        for e in tqdm(range(EPOCHS)):
            accuracy = train(model, optimizer, loss, train_loader, test_loader, e, verbose=5, device=DEVICE)

            if TSNE and e % TSNE_EPOCHS == 0 and e > 0:
                visualize_tsne(mnist_test_data_tsne, nn, e, 0, num_exp)
                visualize_tsne(emnist_test_data_tsne, nn_letters, e, 1, num_exp)
            if CKPT_SAVE > 0 and e > 0 and e % CKPT_SAVE == 0:
                if not os.path.exists('./experiments/'):
                    os.mkdir('./experiments/')
                if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                    os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
                torch.save(model.state_dict(),
                           './experiments/ckpt_{}/ckpt.{}.pth'.format(EXPERIMENT, e))
            if accuracy > 0.968:
                if not os.path.exists('./experiments/'):
                    os.mkdir('./experiments/')
                if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                    os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
                torch.save(model.state_dict(),
                           './experiments/ckpt_{}/ckpt_final.pth'.format(EXPERIMENT, e))
                break

        accuracy_train = test_parity(model, train_loader, device=DEVICE)
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
    parser.add_argument('-o', '--optuna', action='store_true', default=True)
    parser.add_argument('-lr', '--lr', default=0.001)
    parser.add_argument('-er', '--eps_rul', default=0.2)
    args = parser.parse_args()
    if args.eval and len(args.checkpoint) > 1:
        eval(args.checkpoint)
    elif args.optuna:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
        study.optimize(lambda trial: experiment(True, trial), n_trials=100)
        pickle.dump(study, open('./studies/parity_optuna.pkl', 'wb'))
    elif not args.optuna:
        experiment(optimize_hp=False, args=args)
    else:
        print('There is an error in your configuration!. ')



