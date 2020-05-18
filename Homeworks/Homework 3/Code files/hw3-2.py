############################################################
# Created on Fri Aug 24 13:36:53 2018                      #
#                                                          #
# @author: olhartin@asu.edu; updates by sdm                #
#                                                          #
# Program to solve resister network with a voltage source  #
############################################################

import numpy as np                     # needed for arrays
from numpy.linalg import solve         # needed for matrices
from read_netlist import read_netlist  # supplied function to read the netlist
import comp_constants as COMP          # needed for the common constants

# this is the list structure that we'll use to hold components:
# [ Type, Name, i, j, Value ] ; set up an index for each component's property

############################################################
# How large a matrix is needed for netlist? This could     #
# have been calculated at the same time as the netlist was #
# read in but we'll do it here to handle special cases     #
############################################################

def ranknetlist(netlist):              # pass in the netlist

    ### EXTRA STUFF HERE!
    nodes = len(netlist)
    current_max = 0
    voltage_source_counts = 0
    for i in netlist:
        temp = max(i[COMP.I],i[COMP.J])
        if(temp > current_max):
            current_max = temp
        if(i[COMP.TYPE] == COMP.VS):
            voltage_source_counts += 1
    max_node = current_max + voltage_source_counts
    return current_max,max_node

############################################################
# Read the netlist parameters     #
# Get the number of rows to be created for the netlist stamping excluding the node 0 considered as ground by default
# Initialize the following matrices :
# y_add : The stamping matrix
# currents : The combination of independent voltage and current source values as a vector
# voltage : The resultant value after solving the y_add matrix and currents vector
############################################################

netlist = read_netlist()

admittance_node_count, max_nodes = ranknetlist(netlist)

currents = np.zeros((max_nodes), dtype = np.float)
voltages = np.zeros((max_nodes), dtype = np.float)
y_add = np.zeros((max_nodes,max_nodes), dtype = np.float)

############################################################
# Function to stamp the components into the netlist        #
############################################################

def stamper(y_add,netlist,currents,voltages,num_nodes): # pass in the netlist and matrices
    # y_add is the matrix of admittances
    # netlist is the list of lists to analyze
    # currents is the vector of currents
    # voltages is the vector of voltages
    # num_nodes is the number of rows in the admittance matrix allocated for the admittances
    voltage_counter = 0                     #This is a counter variable and not a constant
    for comp in netlist:                            # for each component...
        #print(' comp ', comp)                       # which one are we handling...

        # extract the i,j and fill in the matrix...
        # subtract 1 since node 0 is GND and it isn't included in the matrix

        #voltage_counter = -1    #This counter is used to index the additional rows for the independent voltage source and the Current controlled current source

        #Stamping of resistor values
        if (comp[COMP.TYPE] == COMP.R ):           # a resistor
            i = comp[COMP.I] - 1
            j = comp[COMP.J] - 1
            if (i >= 0):                            # add on the diagonal
                y_add[i,i] += 1.0/comp[COMP.VAL]
            if (j >=0):
                 y_add[j, j] += 1.0 / comp[COMP.VAL]

            if (j>=0 and i>=0):
                y_add[i, j] -= 1.0 / comp[COMP.VAL]
                y_add[j, i] -= 1.0 / comp[COMP.VAL]

        #Stamping of current source values into the current source matrix
        if(comp[COMP.TYPE] == COMP.IS):
            i = comp[COMP.I] - 1
            j = comp[COMP.J] - 1
            if (i >= 0):
                currents[i] -= comp[COMP.VAL]
            if (j >= 0):
                currents[j] += comp[COMP.VAL]

        #Stamping of the voltage source values into the current matrix
        if(comp[COMP.TYPE] == COMP.VS):
            index = num_nodes + voltage_counter
            i = comp[COMP.I] - 1
            j = comp[COMP.J] - 1
            voltage_counter += 1
            currents[index] = comp[COMP.VAL]
            if(i >= 0):
                y_add[index][i] = y_add[i][index] = 1
            if (j >= 0):
                y_add[index][j] = y_add[j][index] = -1
    return num_nodes  # need to update with new value

############################################################
# Start the main program now...                            #
############################################################
temp_num_nodes = stamper(y_add,netlist,currents,voltages,admittance_node_count)

#Perform an error exception to catch the singular matrix errors
try:
    voltages = solve(y_add,currents)
    print(voltages)
except Exception as e:
    print("The folllowing exception was received:")
    print(e)
