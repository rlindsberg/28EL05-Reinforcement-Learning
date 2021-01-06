import numpy as np
import lab1_geo_maze as mz
from matplotlib import pyplot as plt
import csv
from tqdm import tqdm

def define_maze():
    maze = np.zeros((7, 8))
    maze[0:4, 2] = 1  # svart ruta
    maze[5, 1:7] = 1
    maze[5, 1:7] = 1
    maze[6, 4] = 1
    maze[1:4, 5] = 1
    maze[2, 6:8] = 1
    # exit
    maze[6, 5] = 2
    return maze


def draw_maze(maze):
    mz.draw_maze(maze)
    plt.show()


def animate_game_replay(maze, path):
    # Show the shortest path
    mz.animate_solution(maze, path)


# Create an environment maze
def init_game():
    maze = define_maze()
    env = mz.Maze(maze)
    return maze, env


def get_policy(env, horizon):
    # Solve the MDP problem with dynamic programming
    Q, V, policy = mz.dynamic_programming(env, horizon)
    return Q, V, policy


def run_game(env, policy, actual_time_horizon):

    # Simulate the shortest path starting from position A
    method = 'DynProg'
    start = (0, 0, 6, 5)
    path = env.simulate(start, policy, actual_time_horizon, method)
    return path


def get_game_result(path):
    for position in path:
        # x is the horizontal axis and y is vertical axis
        # top-left is 0,0
        player_x = position[0]
        player_y = position[1]
        minotaur_x = position[2]
        minotaur_y = position[3]

        if player_x == minotaur_x and player_y == minotaur_y:
            return "lose"

        if player_x == 6 and player_y == 5:
            return "win"

    return "time"


def main():
    # ### Use this for debugging ###
    # # env.states is a dict that contains all states
    # maze, env = init_game()
    # Q, V, policy = get_policy(env, 20)
    # path = run_game(env, policy)
    # animate_game_replay(maze, path)
    # print(policy)
    # ### Use this for debugging ###

    # ### Use this for getting stats ###
    statistics_array = []

    # step 1. deriving a policy
    training_time_horizon = 30
    maze, env = init_game()
    Q, V, policy = get_policy(env, horizon=training_time_horizon)

    stat = {"win": 0, "lost": 0, "time": 0}

    for game in tqdm(range(10000), desc='game'):
        actual_time_horizon = np.random.geometric(p=1/30)

        # this code snippet takes the first col of policy and expand the 2d array with it
        # not a good solution due to high complexity
        # replaced by changes in lab1_geo_maze.py
        #
        # if time_horizon > training_time_horizon:
        #     policy_first_col = policy[:, 0]  # this is a row array
        #     policy_first_col = policy_first_col.reshape(len(policy_first_col), 1)  # this is a column array
        #     while time_horizon > training_time_horizon:
        #         policy = np.concatenate((policy_first_col, policy), axis=1)
        #
        #     print('done')

        maze, env = init_game()
        path = run_game(env, policy, actual_time_horizon)
        result = get_game_result(path)
        stat[result] += 1

    with open('lab1_c.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        win_rate = stat["win"] / 10000
        lost_rate = stat["lost"] / 10000
        csv_writer.writerow([win_rate, lost_rate])

    # ### Use this for getting stats ###


if __name__ == '__main__':
    main()
