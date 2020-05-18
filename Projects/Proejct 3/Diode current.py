from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as ct
import math

# Provided constants
Is = 1e-9  # Saturation current of the diode
n = 1.7  # Ideality factor
R = 11e3  # Resistor in series with the diode
T = 350  # Temperature coefficient of the diode
q = ct.physical_constants["atomic unit of charge"][0]  # Get the value of the atomic unit of charge
k = ct.physical_constants["Boltzmann constant"][0]  # Boltzmann constant

# Initialize the voltage range for the diode
Vs = np.arange(0.1, 2.5, 0.1)  # Create a voltage range with starting value as 0.1V and end value as 2.5V with a step size of 0.1V
step_size = (2.5 - 0.1) / 0.1
# V = np.ones((1,int(step_size+1)), float)
V = [20 for i in range(int(step_size+1))]  # Creates a list with step_size number of 1000's

# Equation of current passing through the diode
def Idiode(V):
    exp_comp = (V * q) / (n * k * T)
    # print(exp_comp)
    return Is * (np.exp(exp_comp) - 1)

# Define the error equation that is obtained by nodal analysis
def errorFunc(Vd):
    # for i in range(step_size):
    x = (Vd / R) - (Vs / R) + Idiode(Vd)
    return x

# Get the final value of the voltage that gives the least error
diode_voltage = fsolve(errorFunc, V)
diode_current = Idiode(diode_voltage)
print(diode_voltage, diode_current)

current_values = np.zeros((1, len(diode_current)))

# for i in range(len(diode_current)):
#     current_values[i] = math.log(diode_current[i], (10))

print(np.log(diode_current))
plt.plot(Vs, np.log(diode_current), label="Source Voltage vs Diode current")
plt.plot(diode_voltage, np.log(diode_current), label="Diode Voltage vs Diode current")
plt.ylabel("Diode current in log scale")
plt.xlabel("Diode voltage in volts")
plt.legend()
plt.show()



