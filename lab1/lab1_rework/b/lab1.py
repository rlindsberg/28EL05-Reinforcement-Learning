import numpy as np
import lab1_maze as mz
from matplotlib import pyplot as plt


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
    start = (0, 0, 4, 4)
    path = env.simulate(start, policy, method)
    return path


def main():
    maze, env = init_game()
    path = run_game(maze, env, horizon=20)
    animate_game_replay(maze, path)


if __name__ == '__main__':
    main()
