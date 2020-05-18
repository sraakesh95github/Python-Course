#Program to find the Time period of an anharmonic oscillator with the following given params
# m = mass of the pendulum

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

#The upper limit for the integration
m = 1

#Declaring the variables to plot
integral_values = []
x_values = []

#Function definition for the potential of the mass at a height x
def func1(x):
    return (x**6 + (2*(x**2)))

#Multiplicative factor for the integral
const1 = (8*m)**0.5

# Function to be integrated to find the time period of the oscillating pendulum
def func_to_integrate(x,a):
    return (1/((a - (func1(x)))**0.5))

#Values to iterate through to find the integral and plot
iterate_values = np.arange(0.1,2,0.01)

#convert the numpy ndarray to list
list1 = iterate_values.tolist()

#functionality to find the integral
for a_val in list1:
    a_func_value = func1(a_val)
    temp, error = quad(func_to_integrate, 0, a_val, args = (a_func_value))
    integral_values.append(temp*const1)
    x_values.append(a_val)
    
#Plot the integral function
plt.figure("Integration")
plt.plot(x_values, integral_values, label = "Function of time period for an Anharmonic pendulum")
plt.xlabel("Amplitude")
plt.ylabel("Time Period")
plt.title('Time period vs Amplitude plot for Anharmonic oscillator')
        


 
