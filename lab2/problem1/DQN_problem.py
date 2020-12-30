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
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from lab2.problem1.DQN_agent import QAgent, RandomAgent


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.seed(0)
env.reset()

# Parameters
N_episodes = 600  # Number of episodes
discount_factor = 0.99  # Value of the discount factor
n_ep_running_average = 50  # Running average of 50 episodes
n_actions = env.action_space.n  # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
min_eps = 0.01
max_eps = 0.99

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []  # this list contains the total reward per episode
episode_number_of_steps = []  # this list contains the number of steps per episode

# Random agent initialization
agent = QAgent(8, n_actions)
random_agent = RandomAgent(n_actions)

# fill buffer with random exps
state = env.reset()
for i in range(100):
    random_action = random_agent.forward(state)
    next_state, reward, done, _ = env.step(random_action)
    agent.save_exp_to_buffer(state, random_action, reward, next_state, done)
    state = next_state

### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset environment data and initialize variables
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    epsilon = np.maximum(min_eps, max_eps - ((max_eps - min_eps) * i) / ((0.9 * N_episodes) - 1), casting='same_kind')

    if i % 100 == 0:
        print("\n Epsilon is ")
        print(epsilon)
        print("\n")

    while not done:
        # Take a random action
        action = agent.take_action(state, epsilon)

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)

        # Update episode reward
        total_episode_reward += reward

        # Save exp to replay buffer
        agent.save_exp_to_buffer(state, action, reward, next_state, done)

        # learn with discount 0.99
        agent.learn_by_experience(0.99)

        # Update state for next iteration
        state = next_state
        t += 1
        if t == 1000:
            break

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))

    ave_score = running_average(episode_reward_list, n_ep_running_average)[-1]
    if ave_score != 0:
        if ave_score > 100:
            print("i is" + str(i) + '\n')
            torch.save(agent.q_network_local.state_dict(), 'neural-network-1.pth')

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes + 1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes + 1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes + 1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes + 1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
