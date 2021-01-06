from matplotlib import pyplot as plt
import pandas as pd

# plot
df = pd.read_csv('lab1_b.csv', sep=",")
col_times = df['Horizon']
col_money = df['Win']
x = col_times
y = col_money
plt.figure()
plt.scatter(x, y)

plt.title('The probability of escaping')
plt.xlabel('Time horizon')
plt.ylabel('Probability')
plt.show()
