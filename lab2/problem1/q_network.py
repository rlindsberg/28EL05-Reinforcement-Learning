import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    # Q Network model for QAgent
    def __init__(self, state_size, action_size, input_layer_neurons=64, hidden_layer_neurons=64):
        # super class call
        super().__init__()
        self.random_seed = 0

        # torch.nn.Linear(in_features: int, out_features: int, bias: bool = True)
        self.fc1 = nn.Linear(state_size, input_layer_neurons)
        self.fc2 = nn.Linear(input_layer_neurons, hidden_layer_neurons)
        self.out = nn.Linear(hidden_layer_neurons, action_size)

    # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    def forward(self, state):
        # input
        x = state
        x = self.fc1(x)
        # activation
        x = F.relu(x)
        # hidden
        x = self.fc2(x)
        # activation
        x = F.relu(x)
        # output
        x = self.out(x)
        actions_q_values = x
        return actions_q_values


class DeepQNetwork(nn.Module):
    # Deep Q Network model for QAgent
    def __init__(self, state_size, action_size, input_layer_neurons=64, hidden_layer_neurons=128):
        # super class call
        super().__init__()
        self.random_seed = 0

        self.fc1 = nn.Linear(state_size, input_layer_neurons)
        self.fc2 = nn.Linear(input_layer_neurons, hidden_layer_neurons)
        self.fc3 = nn.Linear(hidden_layer_neurons, hidden_layer_neurons)
        self.fc4 = nn.Linear(hidden_layer_neurons, hidden_layer_neurons)
        self.out = nn.Linear(hidden_layer_neurons, action_size)

    def forward(self, state):
        # input
        x = state
        x = self.fc1(x)
        x = F.relu(x)

        # hidden 1
        x = self.fc2(x)
        x = F.relu(x)

        # hidden 2
        x = self.fc2(x)
        x = F.relu(x)

        # hidden 3
        x = self.fc3(x)
        x = F.relu(x)

        # hidden 4
        x = self.fc3(x)
        x = F.relu(x)

        # output
        x = self.out(x)
        actions_q_values = x
        return actions_q_values
