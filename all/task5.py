import numpy as np
import pandas as pd
from comparesignal2 import SignalSamplesAreEqual
import tkinter as tk
from tkinter import Label, Entry, Button




def save_to_file( data, num_lines):
    file_path="C:/Users/DELL/OneDrive/Desktop/DSP TASKS/TASK5/save.txt"
    with open(file_path, 'w') as file:
        for i, value in enumerate(data[:num_lines], start=1):
            file.write(f"{i}\t{value}\n")

def on_submit():
    
    print(input_list)
    num_lines = int(lines_entry.get())
    save_to_file( input_list, num_lines)
    result_label.config(text=f"{num_lines} lines have been saved to the file.")
    





def read_file(file_path) :
    counter=0
    samples=[]
    
    index=[]
    with open(file_path,'r')as file:
        for line in file:
            if counter<=2:
                counter+=1
                continue
            columns=line.split()
            if len(columns)==2:
                samples.append(float(columns[1]))
    return  samples     

    
def Computing_DCT(file_path):
    samples=read_file(file_path)

    res=[]
    N=len(samples)
    K=N
    for k in range(K):
        yk=0
        for n in range(N):
            yk+=samples[n]*np.cos((np.pi/(4*N))*(2*n-1)*(2*k-1))
        res.append(np.sqrt(2/N)*yk)
    return res    
    
def Remove_DC(file_path):
    
    samples=read_file(file_path)
    avg=np.mean(samples)
    new_val=samples-avg
    return new_val



     


# Create the main window
window = tk.Tk()
window.title("Save List to File")

DCT=r"C:\Users\DELL\OneDrive\Desktop\DSP TASKS\TASK5\DCT\DCT_input.txt"
Rem_DC=r"C:\Users\DELL\OneDrive\Desktop\DSP TASKS\TASK5\Remove DC component\DC_component_input.txt"                
expecte_Dct=r"C:\Users\DELL\OneDrive\Desktop\DSP TASKS\TASK5\DCT\DCT_output.txt"
expect_remove=r"C:\Users\DELL\OneDrive\Desktop\DSP TASKS\TASK5\Remove DC component\DC_component_output.txt"

Dct_result=Computing_DCT(DCT) 
Remove_result=Remove_DC(Rem_DC)
  

SignalSamplesAreEqual(expecte_Dct,Dct_result) 
SignalSamplesAreEqual(expect_remove,Remove_result) 

input_list = Dct_result



# Entry for file path

# Entry for the number of lines
lines_label = tk.Label(window, text="Enter the number of lines:")
lines_label.pack()
lines_entry = tk.Entry(window)
lines_entry.pack()

# Button to save the list to a file
save_button = tk.Button(window, text="Save to File", command=on_submit)
save_button.pack()

# Label to display the result
result_label = tk.Label(window, text="")
result_label.pack()

window.mainloop()





   
    
          
            
   
