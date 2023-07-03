import random

import torch



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


class MNISTMultiDigit(torch.nn.Module):
    def __init__(self, nn, epsilon_digits, epsilon_rules, n_digits=10, nn2=None, device='cpu'):
        super(MNISTMultiDigit, self).__init__()
        self.nn = nn
        self.device = device
        if nn2 is not None:
            self.nn2 = nn2
        else:
            self.nn2 = nn

        self.weights_carry = torch.nn.Parameter(torch.randn([n_digits, n_digits, 2, 2]).to(self.device))
        self.weights_carry.requires_grad = True
        self.weights_sum = torch.nn.Parameter(torch.randn([n_digits, n_digits, 2, 10]).to(self.device))
        self.weights_sum.requires_grad = True
        self.epsilon_digits = epsilon_digits
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
            return torch.max(torch.nn.functional.softmax(self.weights_sum, dim=3), dim=3)
        else:
            n_digits = self.weights_sum.shape[0]
            n_carry_symbols = self.weights_sum.shape[2]
            n_output_symbols = self.weights_sum.shape[3]
            random_selection = torch.rand((n_digits, n_digits, n_carry_symbols)) < self.epsilon_rules
            random_selection = random_selection.to(self.device)
            symbol_index_random = torch.randint(n_output_symbols, (n_digits, n_digits, n_carry_symbols))
            symbol_index_random = symbol_index_random.to(self.device)
            _, symbol_index_max = torch.max(self.weights_sum, dim=3)

            chosen_symbols = torch.where(random_selection, symbol_index_random, symbol_index_max)

            truth_values = torch.gather(torch.nn.functional.softmax(self.weights_sum, dim=3),
                                        3, chosen_symbols.view(n_digits, n_digits, n_carry_symbols, 1)).view(n_digits, n_digits, n_carry_symbols)


            return truth_values, chosen_symbols

    def get_rules_matrix_carry(self, eval):
        if eval:
            return torch.max(torch.nn.functional.softmax(self.weights_carry, dim=3), dim=3)
        else:
            n_digits = self.weights_carry.shape[0]
            n_carry_symbols = self.weights_carry.shape[2]
            n_output_symbols = self.weights_carry.shape[3]
            random_selection = torch.rand((n_digits, n_digits, n_carry_symbols)) < self.epsilon_rules
            random_selection = random_selection.to(self.device)
            symbol_index_random = torch.randint(n_output_symbols, (n_digits, n_digits, n_carry_symbols))
            symbol_index_random = symbol_index_random.to(self.device)
            _, symbol_index_max = torch.max(self.weights_carry, dim=3)

            chosen_symbols = torch.where(random_selection, symbol_index_random, symbol_index_max)

            truth_values = torch.gather(torch.nn.functional.softmax(self.weights_carry, dim=3),
                                        3, chosen_symbols.view(n_digits, n_digits, n_carry_symbols, 1)).view(n_digits, n_digits, n_carry_symbols)


            return truth_values, chosen_symbols


    def forward(self, x_t, y_t, eval=False):
        rules_weights_sum, g_matrix_sum = self.get_rules_matrix(eval)
        rules_weights_carry, g_matrix_carry = self.get_rules_matrix_carry(eval)
        truths_list = []

        res_truths = torch.zeros((x_t.shape[0], x_t.shape[1])).to(self.device)
        res_symbols = torch.zeros((x_t.shape[0], x_t.shape[1])).to(self.device)
        previous_carry = 0

        for i in range(x_t.shape[1] - 1, -1, -1):
            x, _ = self.nn(x_t[:, i, :, :].unsqueeze(1))
            y, _ = self.nn2(y_t[:, i, :, :].unsqueeze(1))

            truth_values_x, chosen_symbols_x = self.epsilon_greedy(x, eval)
            truth_values_y, chosen_symbols_y = self.epsilon_greedy(y, eval)

            truths_list += [
                truth_values_x.view(-1, 1),
                truth_values_y.view(-1, 1),
            ]

            # Result (and corresponding truth value) of the prediction of current position
            res_symbols[:, i] = g_matrix_sum[chosen_symbols_x, chosen_symbols_y, previous_carry]
            res_truths[:, i] = torch.min(torch.concat( truths_list +
                                                     [rules_weights_sum[chosen_symbols_x, chosen_symbols_y,
                                                                        previous_carry].view(-1, 1)], dim=1), 1)[0]

            # Carry for the next step
            truths_list.append(rules_weights_carry[chosen_symbols_x, chosen_symbols_y, previous_carry].view(-1, 1))
            previous_carry = g_matrix_carry[chosen_symbols_x, chosen_symbols_y, previous_carry]

        return res_truths, res_symbols