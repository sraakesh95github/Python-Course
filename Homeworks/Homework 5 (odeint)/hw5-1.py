import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

# Define the start and the end points for the function
START = 0
END = 7

# Define the number of points within the range for which the values are to be obtained
NUM_DP = 700

# Calculate the range of values to be given as input
t_step = (END - START) / 700
t_values = np.linspace(START, END, NUM_DP)

###############
# Problem 1
###############

# Define the initial values of y0
y0_prob1 = 1

# Define the function for calculating the derivative
def prob1(y, t):
    return np.cos(t)

# Calculate the integrated result
y_prob1 = odeint(prob1, y0_prob1, t_values)

# Setup the subplots
fig = plt.figure(constrained_layout=False, figsize=(12, 10))
gs = fig.add_gridspec(2, 2)

# PLot the parameters on a graph
f1_ax = fig.add_subplot(gs[0, 0])
f1_ax.plot(t_values, y_prob1)
f1_ax.set_title("Problem 1")
f1_ax.set_xlabel("t values")
f1_ax.set_ylabel("y values")

###############
# Problem 2
###############

# Define the initial value of y0
y0_prob2 = 0

# Calculate the integral function
def prob2(y, t):
    return (-2 * y) + ((2 * t) + 1) * np.exp(-2 * t)

# Get the integral value
y_prob2 = odeint(prob2, y0_prob2, t_values)

# Plot the curve for problem 2
f2_ax = fig.add_subplot(gs[0, 1])
f2_ax.set_title("Problem 2")
f2_ax.plot(t_values, y_prob2)
f2_ax.set_xlabel("t values")
f2_ax.set_ylabel("y values")

##################################################################
# Problem 3

# Consider
# dy = derivative of y wrt t
# dyy = 2nd derivative of y wrt t
# Let 0, 1, 2 indices of y_val represent y, dy, dyy respectively
##################################################################

# Init the y values and its corresponding derivatives
y0_prob3 = [0, -6]

# Define the function for which the integral is to be found out
def prob3(y_in, t_val):
    y = y_in[0]
    dy = y_in[1]
    dyy = (-28 * np.cos(2 * t_val)) + (3 * np.sin(2 * t_val)) - (3 * dy) + y
    return dy, dyy

# Calculate the integral of the given function
y_prob3 = odeint(prob3, y0_prob3, t_values)

# Plot the values calculated for the given function
# Display the plots separately as subplots
f3_ax = fig.add_subplot(gs[1, :])
f3_ax.plot(t_values, y_prob3[:, 0], label="y vs t")
f3_ax.plot(t_values, y_prob3[:, 1], label="y' vs t")
f3_ax.set_title("Problem 3")
f3_ax.set_xlabel("t values")
f3_ax.set_ylabel("y values")
f3_ax.legend()
plt.show()


