import numpy as np

array1 = np.zeros((2,3))

for i in range(2):
    for j in range(3):
        array1[i][j] = i + j

print(array1[0,:])