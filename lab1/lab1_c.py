import numpy as np
import lab1_2_maze as mz
from matplotlib import pyplot as plt

maze = np.zeros((3, 6))
maze[0, 0] = 1 # banks
maze[0, 5] = 1
maze[2, 0] = 1
maze[2, 5] = 1

# exit
#mz.draw_maze(maze)
#plt.show()
numfinishmaze = 0
# Create an environment maze
env = mz.Maze(maze)

# Finite horizon
horizon = 20

gamma = 0.7
epsilon = 0.05
# Solve the MDP problem with dynamic programming
V, policy = mz.value_iteration(env, gamma, epsilon)
#for i in range(100):
# Simulate the shortest path starting from position A
#   method = 'DynProg'
#  start  = (0, 0, 6, 5)
# path = env.simulate(start, policy, method)
#if path[-1][0:2] == (6, 5):
#   numfinishmaze += 1
#print(numfinishmaze)
method = 'ValIter'
start  = (0, 0, 1, 2)
path = env.simulate(start, policy, method)




# Show the shortest path
mz.animate_solution(maze, path)

