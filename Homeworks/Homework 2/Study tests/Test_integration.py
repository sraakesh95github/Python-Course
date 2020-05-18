from scipy.integrate import quad
import numpy as np

#Test program for gaussian quadrature
def func(x):
    return (3*(x**2) + 1)

integral = quad(func,0,2)

print(integral)

pi = np.pi

print(pi)

def f2(x):
    return (np.exp(-x) * np.sin(3*x))

integral2,error2 = quad(f2,0,2*pi)

print(integral2)