import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import seaborn as sns
from ConvTest import ConvTest
from CompareSignal import Compare_Signals

def read_fie(file_path):    
    indices = []
    samples = []
    
    rows_processed = 0   
    with open(file_path, 'r') as file:
        for line in file:
            if rows_processed < 2:
                rows_processed += 1
                continue
        
            columns = line.split()
            if len(columns) == 2:
                index = float(columns[0])
                sample = float(columns[1])
                indices.append(index)
                samples.append(sample)
                
                
    return  indices,samples      
    
def idft(samples):
        d=len(samples)
        X=np.zeros(d, dtype=complex)
        for n in range(d):
            for k in range(d):
                X[n]+=samples[k]*np.exp(2j*math.pi*k*n/d)  
            X[n]*=(1/d)          
        return X
    
    
    
    
def expected_indices(indices1,indices2):
        indices=[]
        start=int(indices1[0])+int(indices2[0])
        end=int(indices1[len(indices1)-1])+int(indices2[len(indices2)-1])
    
        for i in range(start, end + 1):
            indices.append(i)
        return indices    
                

def calculate_dft(samples):

        d=len(samples)
        X=np.zeros(d, dtype=complex)
        for k in range(d):
            for n in range(d):
                X[k]+=samples[n]*np.exp(-2j*math.pi*k*n/d)     
        return X               
               




def append_zeros(signal1,signal2):
    N=len(signal1)+len(signal2)-1
    sample1=np.zeros(N)
    sample2=np.zeros(N)
    for i in range(len(signal1)):sample1[i]=signal1[i]
    for i in range(len(signal2)):sample2[i]=signal2[i]
    
    sample1=calculate_dft(sample1)
    sample2=calculate_dft(sample2)
    
    return sample1,sample2

  

def fast_conv():
    indices1,signal1=read_fie("Input_conv_Sig1.txt")
    indices2,signal2=read_fie("Input_conv_Sig2.txt")
    expected_iduces=expected_indices(indices1,indices2)
  
    sample1,sample2=append_zeros(signal1,signal2)

    result=sample1*sample2
    result=idft(result)
    ConvTest(expected_iduces,result)
   





def fast_corr():
    indices1,signal1=read_fie("Corr_input signal1.txt")
    indices2,signal2=read_fie("Corr_input signal2.txt")
    
    expected_indices=[]
    for i in range(len(indices1)):expected_indices.append(i)
    sample1=calculate_dft(signal1)
    sample2=calculate_dft(signal2)

    for i in range(len(sample1)):
        sample1[i]=sample1[i].real+-1j*sample1[i].imag
 
    result=sample1*sample2
    result=1/(len(sample1))*idft(result)

    
    expected_output="Corr_Output.txt"
    Compare_Signals(expected_output,expected_indices,result)


fast_conv()
fast_corr()
















              