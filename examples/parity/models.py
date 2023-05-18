import random

import torch
from torch.autograd import Variable


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

class ParityFunctionModel(torch.nn.Module):
    def __init__(self, epsilon_rules, device='cpu'):
        super(ParityFunctionModel, self).__init__()
        self.epsilon_greedy = EpsilonGreedy(device)
        self.device = device
        self.weight_initial = torch.nn.Parameter(torch.randn([1, 2]).to(self.device))
        self.weights = torch.nn.Parameter(torch.randn([2, 2, 2]).to(self.device))
        self.weights.requires_grad = True
        self.weight_initial.requires_grad = True
        self.epsilon_rules = epsilon_rules

    def get_rules_matrix(self):
        return self.epsilon_greedy(torch.nn.functional.softmax(self.weights, -1), self.epsilon_rules, True)

    def forward(self, binary_list, eval=False):
        first_truth_value, previous_symbol = self.epsilon_greedy(
            torch.nn.functional.softmax(self.weight_initial, 1).repeat(binary_list.shape[0], 1), self.epsilon_rules,
            eval)
        truth_values_list = [first_truth_value.view(-1)]
        truths, matrix = self.epsilon_greedy(torch.nn.functional.softmax(self.weights, -1), self.epsilon_rules,
                                             eval)  # self.get_rules_matrix(eval)

        for b in range(binary_list.shape[1]):
            previous_symbol = matrix[binary_list[:, b].view(-1, 1), previous_symbol.view(-1, 1)]

            truth_values_list.append(
                truths[torch.squeeze(binary_list[:, b]), torch.squeeze(previous_symbol)].view(-1, 1))

        symbols_truth_values = torch.concat([t.view(-1, 1) for t in truth_values_list], dim=1)
        predictions_truth_values, _ = torch.min(symbols_truth_values, 1)

        return predictions_truth_values, previous_symbol
