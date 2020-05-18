import tkinter
from tkinter import *

root = Tk() #Constructir to init an object of the class Tkinter

root2 = Tk()

#Create a window w with the given label
w = Label(root, text="Hello World!") #root is the parent window
w.pack() #Packs the text within the window

#Create a frame
upperframe = Frame(root)
upperframe.pack()

bottomframe = Frame(root)
bottomframe.pack(side = BOTTOM)

button1 = Button(upperframe, text = "Button1", fg = "Red")
button1.pack(expand = TRUE)

button2 = Button(bottomframe, text = "Button2", fg = "green")
button2.pack(side = RIGHT)

upperframe2 = Frame(root2)
upperframe2.pack()

button21 = Button(upperframe2, text = "button21", fg = 'violet')
button21.pack(fill = X)

mainloop()


