#Develop a GUI based input dialog using tkinter package
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from random import seed
from random import gauss
#Package to generate random numbers

#Define the scale for random numbers
RANDOM_NUMBER_SEED = 1

#Define the number of plots and the number of years for the plots to be generated to
PLOT_MAX_YEARS = 70
NUMBER_OF_PLOTS = 10

#Create a root window
#The constructor Tk() creates an object of the type tkinter class
root = Tk()

#The entries are made to get the following data
#1) Mean Return (%)       This is the average annual return of the investment above inflation
#2) Std Dev Return (%)     This is a measure of the annual volatility of the investment
#3) Yearly Contribution ($)
#4) No. of Years of Contribution
#5) No. of Years to Retirement
#6) Annual Spend in Retirement
e_mean = Entry(root)
e_std = Entry(root)
e_y = Entry(root)
e_nyc = Entry(root)
e_nyr = Entry(root)
e_exp = Entry(root)

#The entries are to be formatted into specific grids within the window
e_mean.grid(row = 0, column = 1)
e_std.grid(row = 2, column = 1)
e_y.grid(row = 4, column = 1)
e_nyc.grid(row = 6, column = 1)
e_nyr.grid(row = 8, column = 1)
e_exp.grid(row = 10, column = 1)

#Provide the labels to the entries
Label(root, text = "Mean Return (%)").grid(row = 0, column = 0)
Label(root, text = "Std Dev Return (%)").grid(row = 2, column = 0)
Label(root, text = "Yearly Contribution ($)").grid(row = 4, column = 0)
Label(root, text = "No. of Years of Contribution").grid(row = 6, column = 0)
Label(root, text = "No. of Years to Retirement").grid(row = 8, column = 0)
Label(root, text = "Annual Spend in Retirement").grid(row = 10, column = 0)


# Gets the mean return %, standard deviation, number of years of contribution and retirement and expenditure as inputs#
# Calculates the white gaussian noise to predict the return of investments during the years of contribution
# Checks for 3 conditions:
#     a. The contribution is made only until the year of contribution
#     b. The expenditure is subtracted only after the year of retirement
# Calculates the average wealth at the time of retirement
# Plots each graph based upon the number of iterations provided in NUMBER_OF_PLOTS
# Plot extends upto 70 years
# THe number of years of contribution or retirement cannot exceed 70 years. It throws an exception


def calculate_return(mean, std, y, nyc, nyr, exp):
    avg_ten_iter = 0
    sum_amt = 0
    avg_rt = 0

#This loop corresponds to the 10 iterations that need to be exceuted
    for j in range(NUMBER_OF_PLOTS):
        amt = 0

#Define the lists for plotting and calculation of the average amount on retirement
        amt_array = []
        years_values = []

#This loop corresponds to the number of years until which the financial returns are expected
        for i in range(PLOT_MAX_YEARS):

#Here the white gaussian noise is generated with a mean of 0 and and standard deviation of 1
            noise = (std / 100) * np.random.randn(1)
            if (i <= nyc):
                if(i <= nyr):
                    yearly_contrib = y
                else:
                    yearly_contrib = y - exp
            else:
                if (i <= nyr):
                    yearly_contrib = 0
                elif (i > nyr):
                    yearly_contrib = -exp
            amt = (amt * (1 + (mean / 100) + noise)) + yearly_contrib
            if(amt <= 0):
                amt_array.append(0)
                years_values.append(i)
                break
            amt_array.append(amt)
            years_values.append(i)

        # PLots for the yearly savings
        plot_name = 'Plot ' + str(j + 1)
        plt.plot(years_values, amt_array, label=plot_name, marker = '+')
        plt.xlabel('Years')
        plt.ylabel('Amount in dollars($)')
        plt.legend()
        plt.title("Wealth post retirement estimator")

        #Summasion of the average on retirement
        avg_rt = avg_rt + float(amt_array[nyr])

    #Calculation for the average during the retirenemnt
    avg_rt = avg_rt / NUMBER_OF_PLOTS

    #Display the average on retirement. WILL BE DISPLAYED ONLY AFTER THE SUBMIT BUTTON HAS BEEN PRESSED
    Label(root, text= format(avg_rt, ",.0f") + " $", font = ('Times New Roman', 12, 'bold')).grid(row=14, column=1)
    plt.show()

#Hover over button reactions
def on_enter_calc(e):
    calc_but['background'] = 'light green'
    calc_but['foreground'] = 'green'

def on_enter_quit(e):
    quit_but['background'] = 'orange'
    quit_but['foreground'] = 'red'

def on_leave(e):
    calc_but['background'] = 'grey'
    calc_but['foreground'] = 'white'
    quit_but['background'] = 'grey'
    quit_but['foreground'] = 'white'

#Button press for test dataset commented
#calc_but = Button(root, text = "Calculate", command = lambda: calculate_return(mean, std, y, nyc, nyr, exp, MAX_VAL_RAN, MIN_VAL_RAN))

#Label to display average amount at retirement
Label(root, text="Average amount at retirement: ", font = ('Times New Roman', 12, 'bold')).grid(row=14, column=0)

#Define the buttons to perform the calculation of the retirement funds and to display the graph or to quit the window
calc_but = Button(root, text = "Calculate", command = lambda: calculate_return(float(e_mean.get()), float(e_std.get()), int(e_y.get()), int(e_nyc.get()), int(e_nyr.get()), int(e_exp.get())), background = 'grey', foreground = 'white')
quit_but = Button(root, text = "Quit", command = root.destroy, background = 'grey', foreground = 'white')
calc_but.bind("<Enter>", on_enter_calc)
calc_but.bind("<Leave>", on_leave)
calc_but.grid(row = 18, column = 0, sticky = N+E+S+W)
quit_but.bind("<Enter>", on_enter_quit)
quit_but.grid(row = 18, column = 1, sticky = N+E+S+W)
quit_but.bind("<Leave>", on_leave)

#Define grid dimensions
root.columnconfigure(0, minsize = 100)
root.columnconfigure(0, minsize = 100)

columns, rows = root.grid_size()

for i in range(rows):
    root.rowconfigure(i, minsize = 7)

#Run the GUI
root.mainloop()