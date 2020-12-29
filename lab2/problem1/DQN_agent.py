# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import random
import copy

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from lab2.problem1.q_network import QNetwork, DeepQNetwork
from lab2.problem1.replay_buffer import ReplayBuffer


def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class QAgent:
    def __init__(self, state_size, action_size):
        self.seed = random.seed(0)
        self.t = 0
        self.state_size = state_size
        self.action_size = action_size

        self.q_network_local = QNetwork(state_size, action_size, 64, 64)
        self.q_network_target = QNetwork(state_size, action_size, 64, 64)
        # learning rate is 10^-4
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=1e-4)

        self.replay_buffer = ReplayBuffer(10000, 64)

    def save_exp_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add_experience(state, action, reward, next_state, done)

    def learn_by_experience(self, gamma):
        if len(self.replay_buffer.double_ended_queue) > 64:
            experiences = self.replay_buffer.get_experiences_tuple()
            states, actions, rewards, next_states, dones = experiences
            q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
            q_targets = rewards + (gamma * q_targets_next * (1 - dones))
            q_expected = self.q_network_local(states).gather(1, actions)
            loss = F.mse_loss(q_expected, q_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.t = (self.t + 1) % 175
        if self.t == 0:
            self.q_network_target = copy.deepcopy(self.q_network_local)

    def take_action(self, state, epsilon):
        if epsilon < np.random.uniform(0, 1):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            # put local network on evaluation mode
            self.q_network_local.eval()
            with torch.no_grad():
                actions_q_values = self.q_network_local(state_tensor)
            # put it back on training mode
            self.q_network_local.train()

            # return greedy action
            return np.argmax(actions_q_values.data.numpy())
        else:
            return np.random.randint(0, self.action_size)


class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''

    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action
