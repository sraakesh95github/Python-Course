################################################################################
# Project 4                                                                    #
# Steve Millman                                                                #
# Run hspice to determine the tphl of a circuit                                #
################################################################################

import numpy as np                  # package needed to read the results file
import subprocess                   # package needed to lauch hspice
from operator import itemgetter     # used to get the least delay elements in
                                    # a list of tuples
import os.path                      # To check if the output file exists
import re

# Declare the constants
FAN_RANGE = 9
NO_OF_INV = 9

################################################################################
# Start the main program here.                                                 #
################################################################################

# launch hspice. Note that both stdout and stderr are captured so
# they do NOT go to the terminal!
# proc = subprocess.Popen(["hspice","InvChain.sp"],
#                           stdout=subprocess.PIPE,
#                           stderr=subprocess.PIPE)
# output, err = proc.communicate()

def generate_fan_inv_lst(fan_max, no_inv):
    # Generate fan list
    fan_list = [i + 2 for i in range(fan_max)]

    # Generate no_inv list
    no_inv_list = [i for i in range(no_inv * 2) if i % 2 != 0]
    
    # The number of inverters to be 1 is obsolete for this problem. So, the number starts from 3 and is an odd number
    no_inv_list.remove(1)
    
    return fan_list, no_inv_list


def gen_replc_str(fan_val, no_of_inv):
    # write fanout
    fanout_sp_stat = ".param fan = " + str(fan_val) + "\n"

    # Specify the number of invertors to be added
    inv_stat = "Xinv1 a b inv M=1\n"
    last_index_val = 0
    for i in range(no_of_inv - 2):
        inv_stat = inv_stat + "Xinv" + str(i+2) + " "  + chr(i + 98) + " " + chr(i + 99) + " inv " + "M=fan**" + str(i + 1) + "\n"
        last_index_val = i + 1

    if(no_of_inv > 1):
        inv_stat = inv_stat + "Xinv" + str(last_index_val + 2) + " " + chr(last_index_val + 98) + " z inv " + "M=fan**" + str(last_index_val + 1)

    replacement_string = fanout_sp_stat + inv_stat
    # print(replacement_string)

    return replacement_string


# def create_sp_file(fan_list, inv_list):
#     # Read the template file
#     f_r = open("InvChainTemplate.sp", "r")
#     template_str = f_r.read()
#
#     # Replace the fan and number of inverters in the template
#     for i in fan_list:
#         for j in inv_list:
#             # Create an hspice file and write the modified contents of fan and number of inverters
#             f_w = open("InvChain.sp", "w+")
#             write_str = re.sub(r'\$.*\$', gen_replc_str(i, j), template_str)
#             f_w.write(write_str)
#             run_hspice()
#
#             # print("File created for fan: " + str(i) + " & Inv count: " + str(j))
#     f_r.close()
#     f_w.close()

##############################################################################################################
# Run hspice to generate the required outputs
##############################################################################################################

def run_hspice():
# launch hspice. Note that both stdout and stderr are captured so
# they do NOT go to the terminal!

    file_name = "InvChain.sp"
    proc = subprocess.Popen(["hspice", file_name],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    output, err = proc.communicate()


# Main program begins here
# Generate the fan and the no_of_inv list
fan_lst, inv_lst = generate_fan_inv_lst(FAN_RANGE, NO_OF_INV)

# Creates the hpice files to be executed
# create_sp_file(fan_lst, inv_lst)

# Read from the template file
f_r = open("InvChainTemplate.sp", "r")
template_str = f_r.read()
f_w = open("InvChain.sp", "w+")

# Delete the InvChain.sp file if such a file exists already


# extract tphl from the output file
print("\n####################################################")
print("\nOutput Delays")
min_delay_list = []
for i in fan_lst:
    print("\n----------Fan: " + str(i) + "------------")
    for j in inv_lst:

        # Create an hspice file and write the modified contents of fan and number of inverters

        write_str = re.sub(r'\$.*\$', gen_replc_str(i, j), template_str)
        f_w.write(write_str)

        # Check if the hspice file exists before it is run
        file_exists = False
        while (not file_exists):
            # simple loop for iteration
            file_exists = os.path.isfile('InvChain.sp')

        # Run the hspice interface to get the output file
        run_hspice()

        # Check if the output file exists before the information is extracted from it
        file_exists = False
        while(not file_exists):
            # simple loop for iteration
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
f_r.close()
f_w.close()

# To find fan with the least delay
print("\n####################################################")    
print("Results...\n")
tup_least_delay = min(min_delay_list, key=itemgetter(0))
print("Least delay: " + str(tup_least_delay[0]))
print("Fan for least delay: " + str(tup_least_delay[1]))
print("#Invertors for least delay: " + str(tup_least_delay[2]))
print("\n####################################################")
