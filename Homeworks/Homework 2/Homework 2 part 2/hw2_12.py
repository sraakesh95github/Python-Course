# Find the value of Stefan Boltzmann constant
# Kb = Boltzmann's constant
# h_bar = planck's constant
# c = Speed of light

import numpy as np
from scipy.integrate import quad

#Define the constants
Kb = 1.380649e-23
h_bar = 1.054571e-34
c = 2.99792e8
const1 = (Kb**4/(4*(np.pi**2)*(c**2)*(h_bar**3)))

#Function to be integrated to get the Stefan Boltzmann constant
def integral_func(z):
    return ((((z/(1-z))**3)/(np.exp((z/(1-z)))-1))*((1/((1-z)**2))))

#Integral function
res, err = quad(integral_func,0,1)

#Print the formated Stefan Boltzmann constant
y = const1 * res
print(format(y, '#.4g'))
