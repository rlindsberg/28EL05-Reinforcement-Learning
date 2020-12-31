import numpy as np
from numpy import arange
import gym
import torch
from math import pi
from tqdm import trange
from lab2.problem1.DQN_agent import QAgent

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_pre_trained_model(path):
    agent = QAgent(state_size=8, action_size=4)
    state_dict = torch.load(path, map_location='cpu')
    agent.q_network_local.load_state_dict(state_dict)
    return agent.q_network_local


def get_action_from_model(state, model):
    pass


def main():
    env = gym.make('LunarLander-v2')
    env.reset()
    state = env.reset()
    y_array = []
    omega_array = []
    max_q_array = []

    model = load_pre_trained_model("neural-network-1.pth")

    done = False
    # s(y, omega) = (0, y, 0, 0, omega, 0, 0, 0)
    state[0] = 0
    state[2:4] = 0
    state[5:8] = 0

    # varying y and omega
    for y in arange(0, 1.5, step=0.0125):
        y_array.append(y)
    for degree in arange(-180, 180, step=3):
        omega = degree / 180 * pi
        omega_array.append(omega)

    # Get action_q_values
    for y in y_array:
        for o in omega_array:
            state[1] = y
            state[4] = o
            action_q_values = model(torch.tensor([state]))
            max_q = np.max(action_q_values.data.numpy())
            # add to arrays for plotting later
            max_q_array.append(max_q)



if __name__ == '__main__':
    main()
