import matplotlib as plt
from random import random
import numpy as np

temp = np.array((10,10), dtype = np.float)

for i in range(10):
    for j in range(10):
        print(temp[i][j])
        temp[i,j] = random()
    plt.plot(temp[i])

plt.show()