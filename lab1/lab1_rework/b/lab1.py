import numpy as np
import lab1_maze as mz
from matplotlib import pyplot as plt
import csv


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


def run_game(maze, env, horizon):
    # Solve the MDP problem with dynamic programming
    V, policy = mz.dynamic_programming(env, horizon)

    # Simulate the shortest path starting from position A
    method = 'DynProg'
    start = (0, 0, 6, 5)
    path = env.simulate(start, policy, method)
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
    statistics_array = []

    # time horizon is 1-20
    for time_horizon in range(1, 21):
        stats = {"time_horizon": time_horizon, "win": 0, "lose": 0, "time": 0}

        # run 100 games
        for game in range(100):
            maze, env = init_game()
            path = run_game(maze, env, horizon=time_horizon)
            result = get_game_result(path)
            stats[result] += 1

        # save results of 100 games to array
        statistics_array.append(stats)

    with open('lab1_b.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for stat in statistics_array:
            win_rate = stat["win"] / 100
            csv_writer.writerow([stat["time_horizon"], win_rate])


if __name__ == '__main__':
    main()
