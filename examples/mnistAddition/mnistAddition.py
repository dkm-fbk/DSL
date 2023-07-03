import os
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


def visual_addition(args):

    EXPERIMENT = 'mnistAddition'
    NUM_EXPERIMENT = 1
    DEVICE = 'cpu'
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 1000
    EPSILON_SYMBOLS = args.eps_sym
    EPSILON_RULES = args.eps_rul
    LR = args.lr
    LOAD_CKPT = args.ckpt



    nn = MNIST_Net().to(DEVICE)
    model = MNISTSumModel(nn, EPSILON_SYMBOLS, EPSILON_RULES, device=DEVICE)
    model.load_state_dict(torch.load(LOAD_CKPT))
    model.eval()
    rules = model.get_rules_matrix(eval=True)[1].squeeze()
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    train_loader, test_loader, mnist_test_data_tsne, mnist_test_data = get_dataset(BATCH_SIZE, BATCH_SIZE_VAL)
    confusion = test_MNIST_visual(nn, mnist_test_data, n_digits=10, device=DEVICE)
    confusion_swapped, p = swap_conf(confusion)
    rules_swapped = swap_rules(rules.squeeze(), p)
    visualize_confusion(confusion, 'Original Confusion Matrix')
    visualize_confusion(confusion_swapped, 'Swapped Confusion Matrix')
    visualize_rules(rules, 'Original Rules Matrix')
    visualize_rules(rules_swapped, 'Swapped Rules Matrix')


    return

def experiment_eval(args=None):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    EXPERIMENT = 'mnistAddition'
    NUM_EXPERIMENT = 5
    DEVICE = 'cpu' #TODO GPU is much faster, but scatter_add_cuda_kernel wasn't implemented in a deterministic way. Thus on GPU exp is not exactly reproducible (issue here: https://discuss.pytorch.org/t/runtimeerror-scatter-add-cuda-kernel-does-not-have-a-deterministic-implementation/132290)
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 1000
    CKPT_SAVE = 50
    EPOCHS = 100

    EPSILON_SYMBOLS = args.eps_sym
    EPSILON_RULES = args.eps_rul
    LR = args.lr



    train_loader, test_loader, mnist_test_data_tsne, mnist_test_data = get_dataset(BATCH_SIZE, BATCH_SIZE_VAL)
    accuracy_train_results = []
    accuracy_test_results = []

    for num_exp in range(NUM_EXPERIMENT):
        print('Starting training number ', num_exp)


        nn = MNIST_Net().to(DEVICE)
        model = MNISTSumModel(nn, EPSILON_SYMBOLS, EPSILON_RULES, device=DEVICE).to(DEVICE)

        optimizer = madgrad.MADGRAD(
            [{'params': list(model.parameters())[:1]}, {'params': list(model.parameters())[1:], 'lr': 1e-3}], lr=LR)
        loss = torch.nn.BCEWithLogitsLoss()
        print('Accuracy in sum task (before training):' + str(test_sum(model, train_loader, DEVICE)))
        # Learning
        for e in tqdm(range(EPOCHS)):
            accuracy = train(model, optimizer, loss, train_loader, test_loader, nn, mnist_test_data, e,
                                 run=num_exp, device=DEVICE)

            if CKPT_SAVE > 0 and e > 0 and e % CKPT_SAVE == 0:
                if not os.path.exists('./experiments/'):
                    os.mkdir('./experiments/')
                if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                    os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
                torch.save(model.state_dict(),
                           './experiments/ckpt_{}/ckpt.{}.pth'.format(EXPERIMENT, e))

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
    model = MNISTSumModel(nn, EPSILON_SYMBOLS, EPSILON_RULES, device=DEVICE).to(DEVICE)

    optimizer = madgrad.MADGRAD(
        [{'params': list(model.parameters())[:1]}, {'params': list(model.parameters())[1:], 'lr': 1e-3}], lr=LR)
    loss = torch.nn.BCEWithLogitsLoss()
    print('Accuracy in sum task (before training):' + str(test_sum(model, train_loader, DEVICE)))
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
    parser.add_argument('-e', '--eval', action='store_true', default=False)
    parser.add_argument('-lr', '--lr', default=0.11639833786002995)
    parser.add_argument('-c', '--ckpt', default='/home/tc94/Desktop/SQ-Learning/experiments/ckpt_mnistAddition/ckpt_final.pth')
    parser.add_argument('-es', '--eps_sym', default=0.2807344052335263)
    parser.add_argument('-er', '--eps_rul', default=0.1077119516324264)
    args = parser.parse_args()
    if args.optuna:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
        study.optimize(experiment_optuna, n_trials=30)
        pickle.dump(study, open('./studies/mnist_addition_optuna.pkl', 'wb'))
    elif not args.optuna and not args.eval:
        experiment_eval(args=args)
    elif args.eval:
        visual_addition(args=args)
    else:
        print('There is an error in your configuration!. ')






