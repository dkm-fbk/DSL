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

class EpsilonGreedy(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(EpsilonGreedy, self).__init__()
        self.device = device

    def forward(self, t, epsilon, eval):
        if eval:
            return torch.max(t, dim=-1, keepdim=True)
        else:
            _, symbol_index_max = torch.max(t, dim=-1)

            n_output_symbols = t.shape[-1]
            random_selection = torch.rand(symbol_index_max.shape) < epsilon
            random_selection = random_selection.to(self.device)
            symbol_index_random = torch.randint(n_output_symbols, symbol_index_max.shape)
            symbol_index_random = symbol_index_random.to(self.device)

            chosen_symbols = torch.where(random_selection, symbol_index_random, symbol_index_max)

            truth_values = torch.gather(
                torch.nn.functional.softmax(t, dim=-1), -1,
                torch.unsqueeze(chosen_symbols.view(symbol_index_max.shape), -1)).view(symbol_index_max.shape)

            return torch.unsqueeze(truth_values, -1), chosen_symbols

class VisualParityFunctionModel(torch.nn.Module):
    def __init__(self, nn, epsilon_digits, epsilon_rules, device='cpu'):
        super(VisualParityFunctionModel, self).__init__()
        self.device = device
        self.epsilon_greedy = EpsilonGreedy(self.device)
        self.nn = nn
        self.weight_initial = torch.nn.Parameter(torch.randn([1, 2]).to(self.device))
        self.weights = torch.nn.Parameter(torch.randn([2, 2, 2]).to(self.device))
        self.weights.requires_grad = True
        self.weight_initial.requires_grad = True
        self.epsilon_digits = epsilon_digits
        self.epsilon_rules = epsilon_rules

    def get_rules_matrix(self):
        return self.epsilon_greedy(torch.nn.functional.softmax(self.weights, -1), self.epsilon_rules, True)

    def forward(self, binary_list, eval=False):
        first_truth_value, previous_symbol = self.epsilon_greedy(
            torch.nn.functional.softmax(self.weight_initial, 1).repeat(binary_list[0].shape[0], 1), self.epsilon_rules,
            eval)
        truth_values_list = [first_truth_value.view(-1)]
        truths, matrix = self.epsilon_greedy(torch.nn.functional.softmax(self.weights, -1), self.epsilon_rules,
                                             eval)  # self.get_rules_matrix(eval)

        for b in binary_list:
            digit_truth_value, digit_symbol = self.epsilon_greedy(self.nn(b)[0], self.epsilon_digits, eval)
            previous_symbol = matrix[digit_symbol.view(-1, 1), previous_symbol.view(-1, 1)]

            truth_values_list.append(truths[torch.squeeze(digit_symbol), torch.squeeze(previous_symbol)].view(-1, 1))
            truth_values_list.append(digit_truth_value.view(-1, 1))

        symbols_truth_values = torch.concat([t.view(-1, 1) for t in truth_values_list], dim=1)
        predictions_truth_values, _ = torch.min(symbols_truth_values, 1)

        return predictions_truth_values, previous_symbol
