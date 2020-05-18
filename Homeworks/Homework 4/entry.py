import tkinter
from tkinter import *

root = Tk()

m1 = Label(root, text = "Text entry 1: ", wraplength = 400).grid(row = 1, column = 0)
m2 = Label(root, text = "Text entry 2: ", wraplength = 400).grid(row = 2, column = 0)

e1 = Entry(root)
e2 = Entry(root)

e1.grid(row = 1, column = 1)
e2.grid(row = 2, column = 1)

def init_entries(e1,e2):
    print("The following were the entered details : \n Entry 1: " + e1.get() + "\n Entry 2: " + e2.get())
    e1.delete(0,END) #Deletes the elemnts within the text box from and to the specified index
    e2.delete(0,END)

b1 = Button(root, text = "Submit", command = lambda: init_entries(e1,e2))
b1.grid(row = 4, column = 0, sticky = W, pady = 4)

exit_button = Button(root, text = "Quit", command = root.destroy).grid(row = 6, column = 0, sticky = W, pady = 4)

root.mainloop()
