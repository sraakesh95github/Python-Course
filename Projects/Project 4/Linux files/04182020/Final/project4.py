################################################################################
# Project 4                                                                    #
# Run hspice to determine the tphl of a circuit                                #
################################################################################

import numpy as np                  # package needed to read the results file
import subprocess                   # package needed to lauch hspice
from operator import itemgetter     # used to get the least delay elements in
                                    # a list of tuples
import os                           # To check if the output file exists
import re                           # Replace the template with the fan and
                                    # number of inverters

# Declare the constants for fan factor and the number of inverters
FAN_RANGE = 9
NO_OF_INV = 9

################################################################################
# fan_max - The max value of fan that needs to be incorporated started from2
# no_inv - Number of inverters that needs to be inserted. Example if given 3
# Creates list of inverters [3, 5, 7] as gives the fan list and the number of
# inverter list as output
################################################################################
def generate_fan_inv_lst(fan_max, no_inv):
    # Generate fan list
    fan_list = [i + 2 for i in range(fan_max)]

    # Generate no_inv list
    no_inv_list = [i for i in range(no_inv * 2) if i % 2 != 0]
    
    # The number of inverters to be 1 is obsolete for this problem. So, the number \
    # starts from 3 and is an odd number
    no_inv_list.remove(1)
    
    return fan_list, no_inv_list

################################################################################
# This function creates an hspice format with the given fan value (fan_val)
# and the number of inverters (no_of_inv)
################################################################################
def gen_replc_str(fan_val, no_of_inv):
    # write fanout
    fanout_sp_stat = ".param fan = " + str(fan_val) + "\n"

    # Specify the number of invertors to be added
    inv_stat = "Xinv1 a b inv M=1\n"
    last_index_val = 0
    for i in range(no_of_inv - 2):
        inv_stat = inv_stat + "Xinv" + str(i+2) + " "  + chr(i + 98) + " " + chr(i + 99)\
                   + " inv " + "M=fan**" + str(i + 1) + "\n"
        last_index_val = i + 1

    if(no_of_inv > 1):
        inv_stat = inv_stat + "Xinv" + str(last_index_val + 2) + " " + chr(last_index_val + 98)\
                   + " z inv " + "M=fan**" + str(last_index_val + 1)

    replacement_string = fanout_sp_stat + inv_stat
    # print(replacement_string)

    return replacement_string


################################################################################
# Run hspice to generate the required output "InvChain.mt0.csv"
################################################################################
def run_hspice():
# launch hspice. Note that both stdout and stderr are captured so
# they do NOT go to the terminal!

    file_name = "InvChain.sp"
    proc = subprocess.Popen(["hspice", file_name],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    output, err = proc.communicate()


################################################################################
# Main program starts here
################################################################################
# Generate the fan and the no_of_inv list
fan_lst, inv_lst = generate_fan_inv_lst(FAN_RANGE, NO_OF_INV)

# Read from the template file
f_r = open("InvChainTemplate.sp", "r")
template_str = f_r.read()

# Delete the output file if file already exists. This output is to be created by
# running the run_hspice() function
if(os.path.isfile('InvChain.mt0.csv')):
        os.remove('InvChain.mt0.csv')
        print("File InvChain.mt0.csv deleted")

# extract tphl from the output file
print("\n####################################################")
print("\nOutput Delays")
print("\nMax fan factor to be tested starting from 2: " + str(FAN_RANGE + 1))
print("\nNumber of inverters to be tested with starting from 3 in odd numbers: " + str((NO_OF_INV * 2) - 1))

# Maintain this list to save the minimum delays for each iteration of fan_lst
# and inv_lst
min_delay_list = []

# Iterates over the various fan values ranging from 2 to FAN_RANGE
for i in fan_lst:
    print("\n----------Fan: " + str(i) + "------------")

    # Iterates over the various fan values ranging from 3 to odd number
    # given a given NO_OF_INV
    for j in inv_lst:

        # Create an hspice file and write the modified contents of fan and number of inverters
        f_w = open("InvChain.sp", "w+")
        write_str = re.sub(r'\$.*\$', gen_replc_str(i, j), template_str)
        f_w.write(write_str)
        f_w.close()        

        # Run the hspice interface to get the output file
        run_hspice()
        
        # Check if the output file exists before the information is extracted from it
        file_exists = False
        while(not file_exists):
            # print("File not found")
            file_exists = os.path.isfile('InvChain.mt0.csv')

        # Read the data from the output file
        data = np.recfromcsv("InvChain.mt0.csv", comments="$", skip_header=3)
 
        # Extract the data from the column "tphl"
        tphl = data["tphl_inv"]
        print("Delay for " + str(j) + " inverters: " + str(tphl) + " seconds")        

        # Form a temporary tuple with the delay, fan and the number of inverters
        temp_tup = (tphl, i, j)
        min_delay_list.append(temp_tup)


# Close the files after their use else they stay in the memory
# CLean the memory and disk of the test files before exit
f_r.close()
os.remove('InvChain.st0')
os.remove('InvChain.tr0')
os.remove('InvChain.pa0')
os.remove('InvChain.ic0')
os.remove('InvChain.mt0.csv')


# To find fan with the least delay and print the result
print("\n####################################################")    
print("Results...\n")
tup_least_delay = min(min_delay_list, key=itemgetter(0))
print("Least delay: " + str(tup_least_delay[0]) + " seconds")
print("Fan for least delay (fan): " + str(tup_least_delay[1]))
print("#Invertors for least delay (N): " + str(tup_least_delay[2]))
print("\n####################################################")
