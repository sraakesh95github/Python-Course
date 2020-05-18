from scipy.integrate import quad
import math as mt
import numpy as np

def func(x):
    return mt.exp(x/(1-x))

num = np.arange(0,1,0.1)
print(num)

for a in np.nditer(num):
    t,t2 = quad(func,0,a)
    print(t)
    
t3,t4 = quad(func,0,1)
print(t3)