import matplotlib.pyplot as plt
import numpy as np

# Create a dataset
X = np.zeros(10)
Y = np.zeros(10)
for i in range(10):
    X[i] = i
    Y[i] = i

plt.figure(1, figsize=(7, 7))
plt.plot(X, Y, color = 'Red')
# plt.show()
plt.title("This is a test plot")
plt.xlabel("Test x")
plt.ylabel("Test y")


plt.savefig("test2.png", dpi = 300, bbox_inches='tight')