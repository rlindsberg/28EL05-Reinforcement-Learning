import numpy as np
import lab1_maze as mz
from matplotlib import pyplot as plt

maze = np.zeros((7, 8))
maze[0:4, 2] = 1 # svart ruta
maze[5, 1:7] = 1
maze[5, 1:7] = 1
maze[6, 4] = 1
maze[1:4, 5] = 1
maze[2, 6:8] = 1
# exit
maze[6, 5] = 2
mz.draw_maze(maze)
plt.show()


# Create an environment maze
env = mz.Maze(maze)

# Finite horizon
horizon = 20
# Solve the MDP problem with dynamic programming
V, policy = mz.dynamic_programming(env, horizon)

# Simulate the shortest path starting from position A
method = 'DynProg'
start  = (0, 0)
path = env.simulate(start, policy, method)

# Show the shortest path
mz.animate_solution(maze, path)
