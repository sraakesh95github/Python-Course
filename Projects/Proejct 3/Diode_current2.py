import numpy as np
import matplotlib as plt
import scipy.constants as ct
from scipy.optimize import leastsq

# Define the constants
q = ct.physical_constants["atomic unit of charge"][0]  # Atomic unit of charge
k = ct.physical_constants["Boltzmann constant"][0]  # Boltzmann costant
T = 375 # Temperature coefficient in Kelvin
A = 1e-8  # Area of cross-section of the diode

# Init the actual values until the least square estimation is started
n = [1.5]  # Ideality factor of the diode
R = [10e3]  # Resistance in series with the diode
phi = [0.8]  #

# Read a text file in python
f = open("DiodeIV.txt", 'r')
# print(f.read())

# Read line by line from the text file
lines = f.readlines()
Vs_lst = []
Id_actual_lst = []

# get the values of Vs and Id from the text file
for line in lines:
    a=line.strip().split()
    # print(a)
    Vs_lst.append(a[0])
    Id_actual_lst.append(a[1])

# Convert the list of strings to an array for calculations on them
Vs_arr = [float(i) for i in Vs_lst]
Id_actual_arr = [float(i) for i in Id_actual_lst]
# print(Vs_arr, Id_actual_arr)

# Init the residual threshold
res_threshold = 1e-6
res_avg = 1

# Diode current equation
def Idiode2(Is, Vs, R, n, I):
    # print(Is)
    # print(Vs)
    # print("Idiode2: " + str(R))
    # print(n)
    # print(I)
    result = Is * (np.exp((((Vs - (I * R)) * q) / (n * k * T))) - 1)
    return result

# Define the equation for the saturation current of the diode
def Isat(phi):
    # print(A, T, phi, q, k, T)
    # print(A, T**2, np.exp(-phi*q/(k*T)))
    return (A * (T**2) * np.exp(-phi*q/(k*T)))

# Define the residual functions
def residual_phi(phi, n, R):
    Is_temp = Isat(phi)
    return np.absolute(Id_actual_arr - Idiode2(Is_temp, Vs_arr, R, n, Id_actual_arr))

def residual_n(n, phi, R):
    Is_temp = Isat(phi)
    return np.absolute(Id_actual_arr - Idiode2(Is_temp, Vs_arr, R, n, Id_actual_arr))

def residual_R(R, n, phi):
    Is_temp = Isat(phi)
    # print("Residual: " + str(R))
    temp = Idiode2(Is_temp, Vs_arr, R, n, Id_actual_arr)
    return np.absolute(Id_actual_arr - temp)

# # Define the residual functions
# def res_n(n):
#     Is_temp = Isat(phi)
#     return act_diode_cur - Idiode(Is_temp, Vs, R, n)
#
# # Define the residual functions
# def res_R(R):
#     Is_temp = Isat(phi)
#     return act_diode_cur - Idiode(Is_temp, Vs, R, n)

# Use a loop until the desired value converges to an accurate value within an error
# threshold
# Use count to keep track of the number of iterations before the convergence
count = 1

while(res_avg > res_threshold):

    # Perform the least squares optimization to get the values of R, phi and n
    R = leastsq(residual_R, R[0], args=(n[0], phi[0]))
    # print("LeastSq (R): " + str(R))
    phi = leastsq(residual_phi, phi[0], args=(n[0], R[0]))
    # print("LeastSq: (phi)" + str(R[0]))
    n = leastsq(residual_n, n[0], args=(phi[0], R[0]))

    # Get the average residual
    res_avg = np.average(np.absolute(residual_phi(phi[0], n[0], R[0])))

    # Tracking parameters
    print("< Iter#: {0} ; phi: {1:.4f} ; n: {2:.4f} ; R: {3:.2f} ; Residual: {4:.3e} >" \
          .format(count, phi[0][0], n[0][0], R[0][0], res_avg))

    count+=1

# Print the final values
print("\n\nEstimated resistance (R): {0:.2f} \u03A9" .format(R[0][0]))
print("Estimated ideality factor (n): {0:.2f}" .format(n[0][0]))
print("Estimated phi: {0:.3f}" .format(phi[0][0]))
print("Number of iterations to converge: {0}".format(count))

# Plotting the log(diode current) vs the diode voltage
plt.plot(Id_actual_arr, Vs_arr)
Is = Isat(phi)
plt.plot(Idiode2(Is, Vs_arr, R, n, ))
