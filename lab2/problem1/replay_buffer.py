from collections import namedtuple, deque
import random
import numpy as np
import torch


class ReplayBuffer:
    # buffer to store experiences
    def __init__(self, buffer_size, batch_size):
        self.seed = random.seed(0)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.double_ended_queue = deque(maxlen=buffer_size)
        self.define_experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add_experience(self, state, action, reward, next_state, done):
        exp = self.define_experience(state, action, reward, next_state, done)
        # todo: check if deque handles max length
        self.double_ended_queue.append(exp)

    def get_experiences_tuple(self):
        # sample batch_size pieces of exps from the queue
        exps = random.sample(self.double_ended_queue, k=self.batch_size)

        # extract from tuple
        np_array_states = np.vstack([exp.state for exp in exps])
        np_array_actions = np.vstack([exp.action for exp in exps])
        np_array_rewards = np.vstack([exp.reward for exp in exps])
        np_array_next_states = np.vstack([exp.state for exp in exps])
        np_array_dones = np.vstack([exp.done for exp in exps]).astype(np.uint8)

        # convert to tensors
        states = torch.from_numpy(np_array_states).float()
        actions = torch.from_numpy(np_array_actions).long()
        rewards = torch.from_numpy(np_array_rewards).float()
        next_states = torch.from_numpy(np_array_next_states).float()
        dones = torch.from_numpy(np_array_dones).float()

        return states, actions, rewards , next_states, dones
