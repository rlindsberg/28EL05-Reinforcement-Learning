import numpy as np
from numpy import arange
import gym
import torch
from math import pi
from lab2.problem1.DQN_agent import QAgent

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def plot(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X, Y = np.meshgrid(X, Y)
    # reshape to columns: 120, rows: dont care
    Z = np.reshape(Z, (-1, 120))

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 240)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.set_xlabel('Height')
    ax.set_ylabel('Angle')
    plt.title('The Q value function of height and angle of the lander')

    plt.savefig("question_f_plot.png")
    plt.show()

def plot_action(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X, Y = np.meshgrid(X, Y)
    # reshape to columns: 120, rows: dont care
    Z = np.reshape(Z, (-1, 120))

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 3)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.set_xlabel('Height')
    ax.set_ylabel('Angle')
    plt.title('The action value function of height and angle of the lander')

    plt.savefig("question_f_plot_2.png")
    plt.show()


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
    max_a_array = []

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
            max_a = np.argmax(action_q_values.data.numpy())

            # add to arrays for plotting later
            max_q_array.append(max_q)
            max_a_array.append(max_a)


    plot(y_array, omega_array, max_q_array)
    plot_action(y_array, omega_array, max_a_array)



if __name__ == '__main__':
    main()
