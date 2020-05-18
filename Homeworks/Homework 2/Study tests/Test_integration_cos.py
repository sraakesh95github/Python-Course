#Test the integration of a cost function
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

start_limit = 0
end_limit = np.pi*2

integral_array = []
x_values = []

def func(x):
    return np.cos(x)

values = np.arange(0,end_limit,0.05)

i = 0
for a in np.nditer(values):
    integral, error = quad(func,0,a)
    integral_array.append(integral)
    x_values.append(a)
    i+=1

fgr, axes = plt.subplots(2)
axes[0].plot(x_values,np.cos(x_values))
axes[1].set_title('Output plot')
axes[1].plot(x_values, integral_array)
plt.show()

