from tqdm import tqdm
from utils import *
import madgrad
from examples.mnistAddition.dataset_generation import *
from examples.mnistAddition.models import *
from examples.mnistAddition.trainer import *
import torch
import random
import numpy as np
import optuna
import pickle
import argparse


def experiment_eval(args=None):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    EXPERIMENT = 'mnistAddition'
    NUM_EXPERIMENT = 1
    DEVICE = 'cpu' #TODO GPU is much faster, but scatter_add_cuda_kernel wasn't implemented in a deterministic way. Thus on GPU exp is not exactly reproducible (issue here: https://discuss.pytorch.org/t/runtimeerror-scatter-add-cuda-kernel-does-not-have-a-deterministic-implementation/132290)
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 1000
    CKPT_SAVE = 500
    EPOCHS = 300

    EPSILON_SYMBOLS = args.eps_sym
    EPSILON_RULES = args.eps_rul
    LR = args.lr
    N = 20





    train_loader, test_loader, mnist_test_data_tsne, mnist_test_data = get_dataset(BATCH_SIZE, BATCH_SIZE_VAL)
    accuracy_train_results = []
    accuracy_test_results = []
    print('Starting training')


    nn = MNIST_Net(N=N).to(DEVICE)
    nn2 = MNIST_Net(N=N).to(DEVICE)
    model = MNISTSumModel(nn, EPSILON_SYMBOLS, EPSILON_RULES, device=DEVICE, nn2=nn2, n_digits=N).to(DEVICE)

    optimizer = madgrad.MADGRAD(
        [{'params': list(model.parameters())[:1]}, {'params': list(model.parameters())[1:], 'lr': 1e-3}], lr=LR)
    loss = torch.nn.BCEWithLogitsLoss()
    print('Accuracy in sum task (before training):' + str(test_sum(model, train_loader, DEVICE)))
    # Learning
    for e in tqdm(range(EPOCHS)):
        accuracy = train(model, optimizer, loss, train_loader, test_loader, nn, mnist_test_data, e, nn2=nn2,
                             run=0, device=DEVICE)


        if CKPT_SAVE > 0 and e > 0 and e % CKPT_SAVE == 0:
            if not os.path.exists('./experiments/'):
                os.mkdir('./experiments/')
            if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
            torch.save(model.state_dict(),
                       './experiments/ckpt_{}/ckpt.{}.pth'.format(EXPERIMENT, e))

    accuracy_test_results = []
    for i in range(10):
        torch.manual_seed(i)
        random.seed(i)
        np.random.seed(i)
        train_loader, test_loader, mnist_test_data_tsne, mnist_test_data = get_dataset(BATCH_SIZE, BATCH_SIZE_VAL)

        accuracy_test = test_sum(model, test_loader, device=DEVICE)
        accuracy_test_results.append(accuracy_test.cpu().numpy())

    acc_test_np = np.array(accuracy_test_results)

    print('Experiment is over. After {} runs on test set we obtained: \n Mean: {}\n Std: {}'.format(NUM_EXPERIMENT,
                                                                                                        np.mean(
                                                                                                            acc_test_np),
                                                                                                        np.std(
                                                                                                            acc_test_np)))

    return np.mean(acc_test_np)


def experiment_optuna(trial=None):
    EXPERIMENT = 'mnistAddition'
    DEVICE = 'cuda:0'
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 1000
    CKPT_SAVE = 50
    EPOCHS = 200
    EPSILON_SYMBOLS = trial.suggest_float('EPSILON_SYMBOLS', 0.0, 0.8)
    EPSILON_RULES = trial.suggest_float('EPSILON_RULES', 0.0, 0.8)
    LR = trial.suggest_float('LR', 5e-4, 5e-1, log=True)




    train_loader, test_loader, mnist_test_data_tsne, mnist_test_data = get_dataset(BATCH_SIZE, BATCH_SIZE_VAL)
    accuracy_train_results = []
    accuracy_test_results = []

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    nn = MNIST_Net().to(DEVICE)
    nn2 = MNIST_Net().to(DEVICE)
    model = MNISTSumModel(nn, EPSILON_SYMBOLS, EPSILON_RULES, nn2=nn2, device=DEVICE).to(DEVICE)

    optimizer = madgrad.MADGRAD(
        [{'params': list(model.parameters())[:1]}, {'params': list(model.parameters())[1:], 'lr': 1e-3}], lr=LR)
    loss = torch.nn.BCEWithLogitsLoss()
    print('Accuracy in sum task (before training):' + str(test_sum(model, train_loader, DEVICE)))
    # Learning
    for e in tqdm(range(EPOCHS)):
        accuracy = train(model, optimizer, loss, train_loader, test_loader, nn, mnist_test_data, e,
                             run=0, device=DEVICE, nn2=nn2)

        if CKPT_SAVE > 0 and e > 0 and e % CKPT_SAVE == 0:
            if not os.path.exists('./experiments/'):
                os.mkdir('./experiments/')
            if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
            torch.save(model.state_dict(),
                       './experiments/ckpt_{}/ckpt.{}.pth'.format(EXPERIMENT, e))

        trial.report(accuracy, e)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    accuracy_train = test_sum(model, train_loader, device=DEVICE)


    print('Experiment is over. Accuracy: {}'.format(accuracy_train))

    return accuracy_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--optuna', action='store_true', default=False)
    parser.add_argument('-lr', '--lr', default=0.0688098324890294)
    parser.add_argument('-es', '--eps_sym', default=0.544765894502977)
    parser.add_argument('-er', '--eps_rul', default=0.05894746523719255)
    args = parser.parse_args()
    if args.optuna:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
        study.optimize(experiment_optuna, n_trials=30)
        pickle.dump(study, open('./studies/mnist_addition_optuna.pkl', 'wb'))
    elif not args.optuna :
        experiment_eval(args=args)
    else:
        print('There is an error in your configuration!. ')



