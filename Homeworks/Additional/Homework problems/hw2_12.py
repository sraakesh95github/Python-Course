#Assign the values for the constants

import numpy as np
from scipy.integrate import quad

Kb = 1.380649e-23
h_bar = 1.054571e-34
c = 2.99792e8
const1 = (Kb**4/(4*(np.pi**2)*(c**2)*(h_bar**3)))

def integral_func(z):
    return ((((z/(1-z))**3)/(np.exp((z/(1-z)))-1))*((1/((1-z)**2))))

res, err = quad(integral_func,0,1)

y = const1 * res
print(format(y, '#.4g'))
