################################################################################
# Project 4                                                                    #
# Steve Millman                                                                #
# Run hspice to determine the tphl of a circuit                                #
################################################################################

import numpy as np                  # package needed to read the results file
import subprocess                   # package needed to lauch hspice
from operator import itemgetter     # used to get the least delay elements in
                                    # a list of tuples

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
    print(fan_list)

    # Generate no_inv list
    no_inv_list = [i for i in range(no_inv * 2) if i % 2 != 0]
    print(no_inv_list)


def gen_replc_str(no_of_inv, fanout_val):
    # write fanout
    fanout_sp_stat = ".param fan = " + str(fanout_val) + "\n"

    # Specify the number of invertors to be added
    inv_stat = "Xinv1 a b inv M=1\n"
    last_index_val = 0
    for i in range(no_of_inv - 2):
        inv_stat = inv_stat + "Xinv1 " + chr(i + 98) + " " + chr(i + 99) + " inv " + "M=fan**" + str(i + 1) + "\n"
        last_index_val = i + 1

    inv_stat = inv_stat + "Xinv1 " + chr(last_index_val + 98) + " z inv " + "M=fan**" + str(last_index_val + 1)

    replacement_string = fanout_sp_stat + inv_stat
    # print(replacement_string)

    return replacement_string

# extract tphl from the output file
min_delay_list = []
for i in fan_list:
    for j in no_inv_list:
        data = np.recfromcsv("InvChainOut_" + str(i) + "_" + str(j) + ".mt0.csv",\
                             comments="$", skip_header=3)
        tphl = data["tphl_inv"]
        # print(tphl)
        temp_tup = (tphl, i, j)
        min_delay_list.append(temp_tup)

# To find fan with the least delay
tup_least_delay = min(min_delay_list, key=itemgetter(0))
print("Least delay: " + str(tup_least_delay[0]))
print("Fan for least delay: " + str(tup_least_delay[1]))
print("#Invertors for least delay: " + str(tup_least_delay[2]))
