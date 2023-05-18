import random

import torch
from torch.autograd import Variable

class MNIST_Net(torch.nn.Module):
    def __init__(self, N=10, channels=1):
        super(MNIST_Net, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 12, 5),
            torch.nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            torch.nn.ReLU(True),
            torch.nn.Conv2d(12, 16, 5),  # 6 12 12 -> 16 8 8
            torch.nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            torch.nn.ReLU(True)
        )
        self.classifier_mid = torch.nn.Sequential(
            torch.nn.Linear(16 * 4 * 4, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU())
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(84, N),
            torch.nn.Softmax(1)
        )
        self.channels = channels

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d):
            print('init conv2, ', m)
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

        if isinstance(m, torch.nn.Linear):
            print('init Linear, ', m)
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier_mid(x)
        x1 = self.classifier(x)
        return x1, x


class MNISTModelMultiop(torch.nn.Module):
    def __init__(self, nn, nn_letters, epsilon_digits, epsilon_letters, epsilon_rules, nn2=None, device='cpu'):
        super(MNISTModelMultiop, self).__init__()
        self.nn = nn
        self.device = device
        if nn2 is not None:
            self.nn2 = nn2
        else:
            self.nn2 = nn
        self.nn_letters = nn_letters
        self.weights = torch.nn.Parameter(torch.randn([10, 10, 4, 82]).to(self.device))
        self.weights.requires_grad = True
        self.epsilon_digits = epsilon_digits
        self.epsilon_letters = epsilon_letters
        self.epsilon_rules = epsilon_rules

    def epsilon_greedy(self, t, eval, dim=1):
        if eval:
            truth_values, chosen_symbols = torch.max(t, dim=dim)
        else:
            random_selection = torch.rand((t.shape[0],)) < self.epsilon_digits
            random_selection = random_selection.to(self.device)
            symbol_index_random = torch.randint(t.shape[1], (t.shape[0],))
            symbol_index_random = symbol_index_random.to(self.device)
            _, symbol_index_max = torch.max(t, dim=dim)

            chosen_symbols = torch.where(random_selection, symbol_index_random, symbol_index_max)
            truth_values = torch.gather(t, dim, chosen_symbols.view(-1, 1))

        return truth_values, chosen_symbols

    def get_rules_matrix(self, eval):
        if eval:
            return torch.max(torch.nn.functional.softmax(self.weights, dim=3), dim=3, keepdim=True)
        else:
            n_digits = self.weights.shape[0]
            n_letters = self.weights.shape[2]
            n_output_symbols = self.weights.shape[3]
            random_selection = torch.rand((n_digits, n_digits, n_letters)) < self.epsilon_rules
            random_selection = random_selection.to(self.device)
            symbol_index_random = torch.randint(n_output_symbols, (n_digits, n_digits, n_letters))
            symbol_index_random = symbol_index_random.to(self.device)
            _, symbol_index_max = torch.max(self.weights, dim=3)

            chosen_symbols = torch.where(random_selection, symbol_index_random, symbol_index_max)

            truth_values = torch.gather(torch.nn.functional.softmax(self.weights, dim=3),
                                        3, chosen_symbols.view(n_digits, n_digits, n_letters, 1)).view(n_digits, n_digits, n_letters)
            return truth_values, chosen_symbols


    def forward(self, x, y, a, eval=False):
        x, _ = self.nn(x)
        y, _ = self.nn2(y)
        a, _ = self.nn_letters(a)

        truth_values_x, chosen_symbols_x = self.epsilon_greedy(x, eval)  # torch.max(x, 1)
        truth_values_y, chosen_symbols_y = self.epsilon_greedy(y, eval)
        truth_values_a, chosen_symbols_a = self.epsilon_greedy(a, eval)
        rules_weights, g_matrix = self.get_rules_matrix(eval)
        symbols_truth_values = torch.concat(
            [rules_weights[chosen_symbols_x, chosen_symbols_y, chosen_symbols_a].view(-1, 1),  # .to(device),
             truth_values_x.view(-1, 1),
             truth_values_y.view(-1, 1),
             truth_values_a.view(-1, 1)], dim=1)
        predictions_truth_values, _ = torch.min(symbols_truth_values, 1)
        return predictions_truth_values, g_matrix[chosen_symbols_x, chosen_symbols_y, chosen_symbols_a]
