#Importing Tkinter package
import tkinter as tk
from tkinter import filedialog
import os
#creating instance
win = tk.Tk()
data_file_path  = filedialog.askopenfilename()
#tittle addition
win.title("Loading of the Data file")

#Disable resize
#win.resizable(False,False)

# Enable resize x disable y dim
#win.resizable(True,False)

#Adding a label
#tk.Label(win, text ="Load  the Data File").grid(column =0,row=0)

#Adding a label that get modifies
a_label = tk.Label(win,text = "DATA File")
a_label.grid(column=0,row=0)

#button click event function
def click_me():
    action.configure(data_file_path)
    
    
#    action.configure(text="Clicked!")
#    a_label.configure(foreground ='blue',text ='A blue label')
#    

#Adding a button
action = tk.Button(win,text="Load Data file",command=click_me)
action.grid(column=0,row=1)







#Start GUI
win.mainloop()
