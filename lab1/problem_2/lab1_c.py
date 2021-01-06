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
x = np.zeros(shape=100)
y = np.zeros(shape=100)
#for i in range(89):
#gamma = (i+1)/100
#print(gamma)
# Finite horizon
horizon = 20 #Necessary even though it's actually an infinite time horizon, so that our simulation can finish
gamma = 0.8 #called lambda in the assignment
epsilon = 0.05
V, policy = mz.value_iteration(env, gamma, epsilon)
#print(V[8])
#print(policy)
#print(V[8])

# x axis values
#x[i] = gamma
# corresponding y axis values
#y[i] = V[8]

# plotting the points
# plotting points as a scatter plot
#plt.scatter(x, y, label= "values", color= "green",
            #marker= "*", s=30)

# naming the x axis
#plt.xlabel('Lambda')
# naming the y axis
#plt.ylabel('Value function result')

# giving a title to my graph
#plt.title('Value function as a function of Lambda at the initial state')
#plt.legend

# function to show the plot
#plt.savefig("2c.1.png")
#plt.show()

method = 'ValIter'
start  = (0, 0, 1, 2) #Corresponds to element number 8 in our statespace
path = env.simulate(start, policy, method, horizon)

# Show the shortest path
mz.animate_solution(maze, path)

