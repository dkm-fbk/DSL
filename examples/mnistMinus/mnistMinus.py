import os
from tqdm import tqdm
from utils import *
import madgrad
from examples.mnistMinus.dataset_generation import *
from examples.mnistMinus.models import *
from examples.mnistMinus.trainer import *
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
    # torch.cuda.manual_seed(0)
    # torch.backends.cudnn.deterministc = True
    # torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = '0'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] =':4096:8'
    # torch.use_deterministic_algorithms(True)
    EXPERIMENT = 'mnistMinus'
    NUM_EXPERIMENT = 5
    DEVICE = 'cpu'  # TODO GPU is much faster, but scatter_add_cuda_kernel wasn't implemented in a deterministic way. Thus on GPU exp is not exactly reproducible (issue here: https://discuss.pytorch.org/t/runtimeerror-scatter-add-cuda-kernel-does-not-have-a-deterministic-implementation/132290)
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 1000
    TSNE = False
    TSNE_EPOCH = 1000
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
        output_dim = 10
        model = MNISTMinusModel(nn, EPSILON_SYMBOLS, EPSILON_RULES, output_dim=output_dim, device=DEVICE).to(DEVICE)
        pretrained_dict = torch.load(LOAD_CKPT)

        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k != 'weights'}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict

        model.load_state_dict(pretrained_dict, strict=False)
        optimizer = madgrad.MADGRAD(
            [{'params': list(model.parameters())[:1]}, {'params': list(model.parameters())[1:], 'lr': 1e-3}], lr=LR)
        loss = torch.nn.BCEWithLogitsLoss()
        print('Accuracy in minus task (before training):' + str(test_sum(model, train_loader, DEVICE)))
        # Learning
        for e in tqdm(range(EPOCHS)):
            accuracy = train(model, optimizer, loss, train_loader, test_loader, nn, mnist_test_data, e,
                                 run=num_exp, device=DEVICE)

            if TSNE and e % TSNE_EPOCHS == 0 and e > 0:
                visualize_tsne(mnist_test_data_tsne, nn, e, 0, num_exp)
            if CKPT_SAVE > 0 and e > 0 and e % CKPT_SAVE == 0:
                if not os.path.exists('./experiments/'):
                    os.mkdir('./experiments/')
                if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                    os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
                torch.save(model.state_dict(),
                           './experiments/ckpt_{}/ckpt.{}.pth'.format(EXPERIMENT, e))
            if accuracy > 0.98:
                if not os.path.exists('./experiments/'):
                    os.mkdir('./experiments/')
                if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                    os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
                torch.save(model.state_dict(),
                           './experiments/ckpt_{}/ckpt_final.pth'.format(EXPERIMENT, e))
                break

        accuracy_test = test_sum(model, test_loader, device=DEVICE)
        accuracy_test_results.append(accuracy_test.cpu().numpy())

    acc_test_np = np.array(accuracy_test_results)

    print('Experiment is over. After {} runs on test set we obtained: \n Mean: {}\n Std: {}'.format(NUM_EXPERIMENT,
                                                                                                    np.mean(acc_test_np),
                                                                                                    np.std(acc_test_np)))
    return np.mean(acc_test_np)

def experiment_optuna(trial, args):
    EXPERIMENT = 'mnistMinus'
    NUM_EXPERIMENT = 1
    DEVICE = 'cuda:0'
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 1000
    TSNE = False
    TSNE_EPOCH = 1000
    CKPT_SAVE = 50
    EPOCHS = 50
    EPSILON_SYMBOLS = trial.suggest_float('EPSILON_SYMBOLS', 0.0, 0.8)
    EPSILON_RULES = trial.suggest_float('EPSILON_RULES', 0.0, 0.8)
    LR = trial.suggest_float('LR', 5e-4, 5e-1, log=True)
    LOAD_CKPT = args.train_checkpoint #use ckpt from mnist addition


    train_loader, test_loader, mnist_test_data_tsne, mnist_test_data = get_dataset(BATCH_SIZE, BATCH_SIZE_VAL)
    accuracy_train_results = []
    accuracy_test_results = []

    for num_exp in range(NUM_EXPERIMENT):
        print('Starting training number ', num_exp)
        torch.manual_seed(num_exp)
        random.seed(num_exp)
        np.random.seed(num_exp)

        nn = MNIST_Net().to(DEVICE)
        output_dim = 10
        model = MNISTMinusModel(nn, EPSILON_SYMBOLS, EPSILON_RULES, output_dim=output_dim, device=DEVICE).to(DEVICE)
        pretrained_dict = torch.load(LOAD_CKPT)

        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k != 'weights'}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict

        model.load_state_dict(pretrained_dict, strict=False)
        optimizer = madgrad.MADGRAD(
            [{'params': list(model.parameters())[:1]}, {'params': list(model.parameters())[1:], 'lr': 1e-3}], lr=LR)
        loss = torch.nn.BCEWithLogitsLoss()
        print('Accuracy in minus task (before training):' + str(test_sum(model, train_loader, DEVICE)))
        # Learning
        for e in tqdm(range(EPOCHS)):
            accuracy = train(model, optimizer, loss, train_loader, test_loader, nn, mnist_test_data, e,
                                 run=num_exp, device=DEVICE)

            if TSNE and e % TSNE_EPOCHS == 0 and e > 0:
                visualize_tsne(mnist_test_data_tsne, nn, e, 0, num_exp)
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

        accuracy_train = test_sum(model, train_loader, device=DEVICE)
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
    parser.add_argument('-tc', '--train_checkpoint', default='./experiments/ckpt_mnistAddition/ckpt_final.pth')
    parser.add_argument('-o', '--optuna', action='store_false', default=True)
    parser.add_argument('-lr', '--lr', default=0.001)
    parser.add_argument('-es', '--eps_sym', default=0.2)
    parser.add_argument('-er', '--eps_rul', default=0.2)
    args = parser.parse_args()
    if args.optuna:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
        study.optimize(lambda trial: experiment(trial, args), n_trials=100)
        pickle.dump(study, open('./studies/mnist_minus_optuna.pkl', 'wb'))
    elif not args.optuna:
        experiment(args=args)
    else:
        print('There is an error in your configuration!. ')



