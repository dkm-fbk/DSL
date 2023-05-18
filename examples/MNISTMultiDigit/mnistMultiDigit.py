import os
from tqdm import tqdm
from utils import *
import madgrad
from examples.MNISTMultiDigit.dataset_generation import *
from examples.MNISTMultiDigit.models import *
from examples.MNISTMultiDigit.trainer import *
import torch
import random
import numpy as np
import optuna
import pickle
import argparse
import dataset_generation_eval
#-e -c ./experiments/ckpt_mnistAdditionMultiDigit/ckpt_final.pth


'''
RES 2
Experiment is over. After 10 runs on test set we obtained: 
 Mean: 0.9505000114440918
 Std: 0.007158905267715454
Experiment is over. After 10 runs on test set (on single digits sum) we obtained: 
 Mean: 0.9799000024795532
 Std: 0.0018622044008225203

RES 4 
Experiment is over. After 10 runs on test set we obtained: 
 Mean: 0.8892000317573547
 Std: 0.0056533110328018665
Experiment is over. After 10 runs on test set (on single digits sum) we obtained: 
 Mean: 0.973039984703064
 Std: 0.0012955243000760674

RES 15 digits 
Experiment is over. After 10 runs on test set we obtained: 
 Mean: 0.6411411166191101
 Std: 0.015149563550949097
Experiment is over. After 10 runs on test set (on single digits sum) we obtained: 
 Mean: 0.9675300717353821
 Std: 0.001492075389251113
 
 
 
 RES 1000
 Experiment is over. After 10 runs on test set we obtained: 
 Mean: 0.0
 Std: 0.0
Experiment is over. After 10 runs on test set (on single digits sum) we obtained: 
 Mean: 0.9652547836303711
 Std: 0.001461292733438313
'''

def eval_N_digits(args, N):

    EXPERIMENT = 'mnistAdditionMultiDigit'
    NUM_EXPERIMENT = 1
    DEVICE = 'cpu'
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 1000
    EPSILON_SYMBOLS = args.eps_sym
    EPSILON_RULES = args.eps_rul
    LR = args.lr
    LOAD_CKPT = args.ckpt



    nn = MNIST_Net().to(DEVICE)
    model = MNISTMultiDigit(nn, EPSILON_SYMBOLS, EPSILON_RULES, 10, device=DEVICE)
    model.load_state_dict(torch.load(LOAD_CKPT))
    model.eval()
    accuracy_test_results = []
    accuracy_test_results_single = []
    for e in range(10):
        torch.manual_seed(e)
        random.seed(e)
        np.random.seed(e)
        train_loader_complex, test_loader_complex, mnist_train_data, mnist_test_data = dataset_generation_eval.get_dataset_eval(
            BATCH_SIZE, BATCH_SIZE_VAL, N)
        start = time.time()
        accuracy_test = test_sum_multi(model, test_loader_complex, device=DEVICE)
        print(time.time() - start)
        start = time.time()
        accuracy_test_single = test_sum_multi_single(model, test_loader_complex, device=DEVICE)
        print(time.time() - start)
        accuracy_test_results.append(accuracy_test.cpu().numpy())
        accuracy_test_results_single.append(accuracy_test_single.cpu().numpy())

    acc_test_np = np.array(accuracy_test_results)
    acc_test_single_np = np.array(accuracy_test_results_single)

    print(
        'Experiment is over. After {} runs on test set we obtained: \n Mean: {}\n Std: {}'.format(10,
                                                                                                      np.mean(
                                                                                                          acc_test_np),
                                                                                                      np.std(
                                                                                                          acc_test_np)))
    print(
        'Experiment is over. After {} runs on test set (on single digits sum) we obtained: \n Mean: {}\n Std: {}'.format(10,
                                                                                                  np.mean(
                                                                                                      acc_test_single_np),
                                                                                                  np.std(
                                                                                                      acc_test_single_np)))
    return


def experiment_eval(args):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    EXPERIMENT = 'mnistAdditionMultiDigit'
    NUM_EXPERIMENT = 1
    DEVICE = 'cuda:0'
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 1000
    TSNE = False
    TSNE_EPOCH = 1000
    CKPT_SAVE = 50
    EPOCHS = 400

    EPSILON_SYMBOLS = args.eps_sym
    EPSILON_RULES = args.eps_rul
    LR = args.lr



    train_loader_simple, test_loader_simple, train_loader_complex, test_loader_complex, mnist_train_data, mnist_test_data = get_dataset(BATCH_SIZE, BATCH_SIZE_VAL)


    nn = MNIST_Net().to(DEVICE)
    model = MNISTMultiDigit(nn, EPSILON_SYMBOLS, EPSILON_RULES, 10, device=DEVICE)
    optimizer = madgrad.MADGRAD(
        [{'params': list(model.parameters())[:1]}, {'params': list(model.parameters())[1:], 'lr': 1e-3}], lr=LR)
    loss = torch.nn.BCEWithLogitsLoss()
    print('Learning simple problem (Sum one digit without padding)')
    for e in tqdm(range(EPOCHS)):
        accuracy = train(model, optimizer, loss, train_loader_simple, test_loader_simple, nn,
                         mnist_test_data, e, run=0, device=DEVICE)

    print('End of simple problem. Going more complex (Sum 2 digits + padding)')
    for e in tqdm(range(EPOCHS)):
        accuracy = train(model, optimizer, loss, train_loader_complex, test_loader_complex, nn,
                         mnist_test_data, e, run=0, device=DEVICE)
        # if s.tsne > 0 and e % s.tsne == 0:
        #    visualize_tsne(mnist_test_data_tsne, nn, e, 0, num_exp, writer)
        if CKPT_SAVE > 0 and e > 0 and e % CKPT_SAVE == 0:
            if not os.path.exists('./experiments/'):
                os.mkdir('./experiments/')
            if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
            torch.save(model.state_dict(),
                       './experiments/ckpt_{}/ckpt.{}.pth'.format(EXPERIMENT, e))
        if accuracy > 0.999:
            print('accuracy > 0.99')
            if not os.path.exists('./experiments/'):
                os.mkdir('./experiments/')
            if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
            torch.save(model.state_dict(),
                       './experiments/ckpt_{}/ckpt_final.pth'.format(EXPERIMENT, e))
            break
    accuracy_test_results = []
    for e in range(10):
        accuracy_test = test_sum_multi(model, test_loader_complex, device=DEVICE)
        accuracy_test_results.append(accuracy_test.cpu().numpy())

    acc_test_np = np.array(accuracy_test_results)

    print(
        'Experiment is over. After {} runs on test set we obtained: \n Mean: {}\n Std: {}'.format(10,
                                                                                                      np.mean(
                                                                                                          acc_test_np),
                                                                                                      np.std(
                                                                                                          acc_test_np)))
    return np.mean(acc_test_np)


def experiment(trial=None, args=None):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    EXPERIMENT = 'mnistAdditionMultiDigit'
    NUM_EXPERIMENT = 1
    DEVICE = 'cuda:0'
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 1000
    TSNE = False
    TSNE_EPOCH = 1000
    CKPT_SAVE = 50
    EPOCHS = 300
    EPSILON_SYMBOLS = trial.suggest_float('EPSILON_SYMBOLS', 0.0, 0.8)
    EPSILON_RULES = trial.suggest_float('EPSILON_RULES', 0.0, 0.8)
    LR = trial.suggest_float('LR', 5e-4, 5e-1, log=True)



    train_loader_simple, test_loader_simple, train_loader_complex, test_loader_complex, mnist_train_data, mnist_test_data = get_dataset(BATCH_SIZE, BATCH_SIZE_VAL)
    accuracy_train_results = []
    accuracy_test_results = []



    nn = MNIST_Net().to(DEVICE)
    model = MNISTMultiDigit(nn, EPSILON_SYMBOLS, EPSILON_RULES, 10, device=DEVICE)
    optimizer = madgrad.MADGRAD(
        [{'params': list(model.parameters())[:1]}, {'params': list(model.parameters())[1:], 'lr': 1e-3}], lr=LR)
    loss = torch.nn.BCEWithLogitsLoss()
    print('Learning simple problem (Sum one digit without padding)')
    for e in tqdm(range(EPOCHS)):
        accuracy = train(model, optimizer, loss, train_loader_simple, test_loader_simple, nn,
                         mnist_test_data, e, run=0, device=DEVICE)
        if accuracy > 0.97:
            print('accuracy > 0.97')
            break

    print('End of simple problem. Going more complex (Sum 2 digits + padding)')
    for e in tqdm(range(EPOCHS)):
        accuracy = train(model, optimizer, loss, train_loader_complex, test_loader_complex, nn,
                         mnist_test_data, e, run=0, device=DEVICE)
        trial.report(accuracy, e)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
        if CKPT_SAVE > 0 and e > 0 and e % CKPT_SAVE == 0:
            if not os.path.exists('./experiments/'):
                os.mkdir('./experiments/')
            if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
            torch.save(model.state_dict(),
                       './experiments/ckpt_{}/ckpt.{}.pth'.format(EXPERIMENT, e))
        if accuracy > 0.98:
            print('accuracy > 0.98')
            if not os.path.exists('./experiments/'):
                os.mkdir('./experiments/')
            if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
            torch.save(model.state_dict(),
                       './experiments/ckpt_{}/ckpt_final.pth'.format(EXPERIMENT, e))
            break

    accuracy_train = test_sum_multi(model, train_loader_complex, device=DEVICE)
    print('Accuracy train: {}'.format(accuracy_train))

    return accuracy_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval',
                        action='store_true', default=False)
    parser.add_argument('-o', '--optuna', action='store_true', default=False)
    parser.add_argument('-lr', '--lr', default=0.1)
    parser.add_argument('-c', '--ckpt', default='')
    parser.add_argument('-es', '--eps_sym', default=0.0)
    parser.add_argument('-er', '--eps_rul', default=0.2)
    args = parser.parse_args()
    if args.optuna and not args.eval:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
        study.optimize(lambda trial: experiment(trial) , n_trials=100)
        pickle.dump(study, open('./studies/mnist_multi_digit_optuna.pkl', 'wb'))
    elif not args.optuna and not args.eval:
        experiment_eval(args)
    elif args.eval and len(args.ckpt) > 0:
        eval_N_digits(args, 2)
    else:
        print('There is an error in your configuration!. ')






