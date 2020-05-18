#Program to find the integral of anharmonic oscillation using Python's Gaussian Quadrature method
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

#The upper limit for the integration
a = 2
m = 1

#Declaring the variables to plot
integral_values = [0]
x_values = [0]

#Function definition
def func1(x):
    return ((x**6) + (2*(x**2)))

def func_to_integrate(a,x):
    return (((8*m)**0.5)/(((func1(a)) + (func1(x)))**0.5))

#Values to iterate through to find the integral
iterate_values = np.arange(0,2.01,0.01)

#functionality to find the integral
i = 0
for a_val in np.nditer(iterate_values):
    temp, error = quad(func_to_integrate, 0, a_val, args = (a_val))
    integral_values.append(temp)
    x_values.append(a_val)
    i += 1
    
#Plot the integral function
plt.figure("Integration")
plt.plot(x_values, integral_values, label = "Function of time period for an Anharmonic pendulum", marker = "+")
plt.xlabel("Time")
plt.ylabel("Time Period")
        


 