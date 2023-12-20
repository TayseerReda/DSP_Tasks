import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from comparesignals import SignalSamplesAreEqual
import comparesignals
from comparesignal2 import SignalSamplesAreEqual
from QuanTest1 import QuantizationTest1
from QuanTest2 import QuantizationTest2
import math
from signalcompare import SignalComapreAmplitude
from signalcompare import SignalComaprePhaseShift
from Shift_Fold_Signal import Shift_Fold_Signal
from DerivativeSignal import DerivativeSignal
import random
import seaborn as sns
from ConvTest import ConvTest
from CompareSignal import Compare_Signals

def read_file(file_path):    
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
                index = float(columns[0].rstrip(',').rstrip('f'))
                sample = float(columns[1].rstrip('f'))
                indices.append(index)
                samples.append(sample)
                
                
    return  indices,samples    


#-----------first task------------
def first1():
    # Function to display signal samples
    def display_signal(samples, title):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(samples)
        plt.title(f'{title} (Continuous)')

        plt.subplot(1, 2, 2)
        plt.stem(samples, use_line_collection=True)
        plt.title(f'{title} (Discrete)')

        plt.tight_layout()
        plt.show()

    # Function to generate sinusoidal or cosinusoidal signals
    def generate_signal(signal_type, A, theta, f, Fs):
        t = np.linspace(0, 1, int(Fs), endpoint=False)
        if signal_type == 'sine':
            signal = A * np.sin(2 * np.pi * f * t + theta)
        elif signal_type == 'cosine':
            signal = A * np.cos(2 * np.pi * f * t + theta)
        return signal

    # Function to generate the signal
    def generate_and_display_signal():
        signal_type = signal_type_var.get()
        A = float(A_var.get())
        theta = float(theta_var.get())
        f = float(f_var.get())
        Fs = float(Fs_var.get())

        if Fs < 2 * f  :
            result_label.config(text="Warning: Sampling frequency is below Nyquist rate.")
        elif  Fs == 0:
            result_label.config(text="Warning: Sampling frequency is Zero.")

        else:
            result_label.config(text="")
            generated_signal = generate_signal(signal_type, A, theta, f, Fs)
            display_signal(generated_signal, "Generated Signal")
            expected_output_file = "SinOutput.txt"
           # expected_output_file = "CosOutput.txt"
            comparesignals.SignalSamplesAreEqual(expected_output_file, Fs, generated_signal)

    # Create the main GUI window
    root = tk.Tk()
    root.title("Signal Generator")
    root.configure(bg="MistyRose2")

    # Create and place GUI elements
    signal_type_label = tk.Label(root, text="Signal Type:")
    signal_type_label.pack()

    signal_type_var = tk.StringVar(value="sine")
    signal_type_menu = tk.OptionMenu(root, signal_type_var, "sine", "cosine")
    signal_type_menu.pack()

    A_label = tk.Label(root, text="Amplitude (A):")
    A_label.pack()
    A_var = tk.Entry(root)
    A_var.pack()

    theta_label = tk.Label(root, text="Phase shift (theta in radians):")
    theta_label.pack()
    theta_var = tk.Entry(root)
    theta_var.pack()

    f_label = tk.Label(root, text="Analog frequency (f):")
    f_label.pack()
    f_var = tk.Entry(root)
    f_var.pack()

    Fs_label = tk.Label(root, text="Sampling frequency (Fs):")
    Fs_label.pack()
    Fs_var = tk.Entry(root)
    Fs_var.pack()

    generate_button = tk.Button(root, text="Generate and Display Signal", command=generate_and_display_signal)
    generate_button.pack()

    result_label = tk.Label(root, text="")
    result_label.pack()

    root.mainloop()
    
def first2():
    file_path = "signal1.txt"
    indices,samples=read_file(file_path)
        
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(indices, samples)
    plt.title('Continuous Representation')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
        
    plt.subplot(122)
    plt.stem(indices, samples, use_line_collection=True)
    plt.title('Discrete Representation')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
        
    plt.tight_layout()
    plt.show()
    
#-----------second task----------
def second():
    def read_signal_file(file_path):
        indices = []
        samples = []

        with open(file_path, 'r') as file:
            for line in file:
                columns = line.split()
                if len(columns) == 2:
                    index = float(columns[0])
                    sample = float(columns[1])
                    indices.append(index)
                    samples.append(sample)

        return np.array(indices), np.array(samples)

    # Function to perform signal operations
    def perform_operation():
        choice = choice_var.get()
        signalChoice=signalChoice_var.get()
        choice1 = choice1_var.get()
        choice2 = choice2_var.get()
        choice3 = choice3_var.get()
        constant = constant_var.get()
        norm = norm_var.get()
        
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.title("Resulting Signal")

        if choice == "Addition":
            if signalChoice=="2":
                result_signal = signals[choice1] + signals[choice2]
            elif signalChoice=="3":
                result_signal = signals[choice1] + signals[choice2] + signals[choice3]
            plt.plot(result_signal)
            plt.show()
        elif choice == "Subtraction":
            result_signal = signals[choice1] - signals[choice2]
            plt.plot(result_signal)
            plt.show()
        elif choice == "Multiplication":
            result_signal = signals[choice1] * constant
            plt.plot(result_signal)
            plt.show()
        elif choice == "Squaring":
            result_signal = signals[choice1] ** 2
            plt.plot(result_signal)
            plt.show()
        elif choice == "Shifting":
            result_signal = indices4-constant
            plt.plot(result_signal, signals[choice1])
            plt.show()
        elif choice == "Normalization":
            min_value = signals[choice1].min()
            max_value = signals[choice1].max()

            if norm == 1:
                result_signal = (2 * (signals[choice1] - min_value) / (max_value - min_value)) - 1
            elif norm == 2:
                result_signal = (signals[choice1] - min_value) / (max_value - min_value)
                
            plt.plot(result_signal)
            plt.show()
        elif choice == "Accumulation":
            acc_signal = signals[choice1]
            res=np.zeros_like(acc_signal)
            res[0]=acc_signal[0]
            for i in range(len(acc_signal)):
                if i!=0:
                    res[i]=res[i-1]+acc_signal[i]
            result_signal=res
            plt.plot(result_signal)
            plt.show()
        
        result_text.config(text="Resulting signal:\n" + str(result_signal))

    # Read the three signal files
    indices1, signal1 = read_signal_file("Signal1.txt")
    indices2, signal2 = read_signal_file("Signal2.txt")
    indices3, signal3 = read_signal_file("signal3.txt")
    indices4, signal4 = read_signal_file("Input Shifting.txt")

    signals = {
        "Signal 1": signal1,
        "Signal 2": signal2,
        "Signal 3": signal3,
        "Signal 4": signal4
    }

    # Create the main GUI window
    root = tk.Tk()
    root.title("Signal Processing GUI")
    root.configure(bg="MistyRose2")

    # Create and configure GUI components
    choice_label = tk.Label(root, text="Choose an operation:")
    choice_label.pack()

    choices = ["Addition", "Subtraction", "Multiplication", "Squaring", "Shifting", "Normalization", "Accumulation"]
    choice_var = tk.StringVar(root)
    choice_var.set(choices[0])
    choice_menu = tk.OptionMenu(root, choice_var, *choices)
    choice_menu.pack()

    signalChoices_label= tk.Label(root, text="Choose number of signals:")
    signalChoices_label.pack()
    signalChoices = ["1","2","3"]
    signalChoice_var = tk.StringVar(root)
    signalChoice_var.set(signalChoices[0])
    choice_menu = tk.OptionMenu(root, signalChoice_var, *signalChoices)
    choice_menu.pack()

    choice1_label = tk.Label(root, text="Select the first signal:")
    choice1_label.pack()
    choice1_var = tk.StringVar(root)
    choice1_var.set("Signal 1")
    choice1_menu = tk.OptionMenu(root, choice1_var, *signals.keys())
    choice1_menu.pack()

    choice2_label = tk.Label(root, text="Select the second signal:")
    choice2_label.pack()
    choice2_var = tk.StringVar(root)
    choice2_var.set("Signal 2")
    choice2_menu = tk.OptionMenu(root, choice2_var, *signals.keys())
    choice2_menu.pack()

    choice2_label = tk.Label(root, text="Select the third signal:")
    choice2_label.pack()
    choice3_var = tk.StringVar(root)
    choice3_var.set("Signal 3")
    choice2_menu = tk.OptionMenu(root, choice3_var, *signals.keys())
    choice2_menu.pack()

    constant_label = tk.Label(root, text="Enter a constant:")
    constant_label.pack()
    constant_var = tk.DoubleVar()
    constant_entry = tk.Entry(root, textvariable=constant_var)
    constant_entry.pack()

    norm_label = tk.Label(root, text="Enter a normalization option (1 or 2):")
    norm_label.pack()
    norm_var = tk.IntVar()
    norm_var.set(1)
    norm_entry = tk.Entry(root, textvariable=norm_var)
    norm_entry.pack()

    perform_button = tk.Button(root, text="Perform Operation", command=perform_operation)
    perform_button.pack()

    result_text = tk.Label(root, text="Resulting signal:")
    result_text.pack()

    root.mainloop()

#--------third task--------
# def third():

def third():
    def quantize_signal(num_levels=None, num_bits=None):
            # file_path = "Quan1_input.txt"
            file_path = "Quan2_input.txt"

            signal = []

            rows_processed = 0

            with open(file_path, 'r') as file:
                for line in file:
                    if rows_processed < 2:
                        rows_processed += 1
                        continue

                    columns = line.split()
                    if len(columns) == 2:
                        sample = float(columns[1])
                        signal.append(sample)

            signal = np.array(signal)  # Convert to a NumPy array
            

            
            if num_levels is not None:
                levels = num_levels
                num_bits = int(math.log2(levels))
            elif num_bits is not None:
                levels = 2 ** num_bits

            min_signal, max_signal = min(signal), max(signal)
            step_size = (max_signal - min_signal) / levels

            intervals = []
            m = min_signal

            for i in range(levels):
                t = [np.round(m , 2), np.round(m + step_size, 2)]
                intervals.append(t)
                m = m + step_size

            quantization = []

            for interval in intervals:
                quantization.append((interval[1] + interval[0]) / 2)

            quantization=np.round( quantization, 3)
            quantized_signal = []
            idx=[]

            for i in signal:
                found = False  # Track if a suitable interval is found
                for j in intervals:
                    if i >= j[0] and i <= j[1]:
                        quantized_signal.append(quantization[intervals.index(j)])
                        idx.append(intervals.index(j)+1)
                       
                        found = True
                        break
                if not found:
                    # If no suitable interval is found, append a default value (e.g., 0)
                    quantized_signal.append(0)
            quantized_signal = np.round(quantized_signal, 3)
            quantization_error = quantized_signal-signal

            # Simple binary encoding
            binary_encoding = [bin(i)[2:].zfill(num_bits) for i in range(levels)]

            encoded_signal = []
            for s in quantized_signal:
                index = int((s - min_signal) // step_size)
                if index < 0:
                    index = 0
                elif index >= levels:
                    index = levels - 1
                encoded_signal.append(binary_encoding[index])

            return encoded_signal, quantized_signal, quantization_error,idx,quantization
    def quantize_signal_from_gui():
            choice = quantization_choice.get().strip().lower()
    
            if choice == "levels":
                num_levels = int(levels_entry.get())
                result = quantize_signal(num_levels=num_levels)
            elif choice == "bits":
                num_bits = int(bits_entry.get())
                result = quantize_signal(num_bits=num_bits)
            else:
                result = None
    
            if result:
                encoded_signal, quantized_signal, quantization_error,idx,signal= result
                result_label.config(text=" interval index: {}\nEncoded Signal: {}\nQuantized Signal: {}\nQuantization Error: {}\n My Test: {}".format(idx,encoded_signal, quantized_signal, quantization_error,signal))
                plt.plot(quantized_signal)
                plt.title("Quantized Signal")
                plt.show()
            else:
                result_label.config(text="Invalid choice. Please choose 'Levels' or 'Bits.")
            expected_output_file = "Quan2_Out.txt"
            QuantizationTest2(expected_output_file,idx,encoded_signal, quantized_signal,quantization_error)
    
            expected_output_file1 = "Quan1_Out.txt"
            QuantizationTest1(expected_output_file1, encoded_signal, quantized_signal)
    
        # Create the GUI window
    root3 = tk.Tk()
    root3.title("Signal Quantization")
    root3.geometry("400x300")
    root3.configure(bg="MistyRose2")
    
    # Quantization Choice
    quantization_label = tk.Label(root3, text="Choose quantization type (Levels or Bits):")
    quantization_label.pack()
    quantization_choice = ttk.Combobox(root3, values=["Levels", "Bits"])
    quantization_choice.set("Levels")
    quantization_choice.pack()
    
        # Levels Entry
    levels_label = tk.Label(root3, text="Enter the number of levels:")
    levels_label.pack()
    levels_entry = tk.Entry(root3)
    levels_entry.pack()
    
        # Bits Entry
    bits_label = tk.Label(root3, text="Enter the number of bits:")
    bits_label.pack()
    bits_entry = tk.Entry(root3)
    bits_entry.pack()
    
        # Quantize Button
    quantize_button = tk.Button(root3, text="Quantize", command=quantize_signal_from_gui)
    quantize_button.pack()
    
        # Result Label
    result_label = tk.Label(root3, text="")
    result_label.pack()
    
        # Start the GUI
    root3.mainloop()
    
#---------fourth task--------
def fourth():
    def clean_number(number_str):
       return float(number_str.replace('f', '').strip())
                
    def read_file2(file_path):
            complex_array = []
            rows_processed = 0
        
            with open(file_path, 'r') as file:
                for line in file:
                    if rows_processed <= 2:
                        rows_processed += 1
                        continue
                    number_str = line.split(',')
                    A = clean_number(number_str[0])
                    Theta = clean_number(number_str[1])
                    real=A*math.cos(Theta)
                    imag=A*math.sin(Theta)
                    complex_array.append(complex(real,imag))
        
            return complex_array
        
    
    def calculate_dft():
            file_path = "input_Signal_DFT.txt"   
            indices,samples=read_file(file_path) 
            
            Fs=int(Fs_entry.get())
            d=len(samples)
            X=np.zeros(d, dtype=complex)
            
            for k in range(d):
                for n in range(d):
                    X[k]+=samples[n]*np.exp(-2j*math.pi*k*n/d)
            
            frequency_indices=[]
            for j in range(d):
                # print(j)
                frequency_indices.append((2 * np.pi * Fs / d) * (j + 1))
                print(Fs)
            
            print(frequency_indices)
            def calculate_amplitude_and_phase(X):
                amplitude = np.abs(X)
                amplitude=np.round(amplitude,12)
                phase = np.angle(X)
            
                return amplitude, phase
            
            amplitude, phase = calculate_amplitude_and_phase(X)
            
            # Save amplitude and phase to a text file
            output_file = "frequency_components.txt"
            with open(output_file, 'w') as file:
                file.write("0\n1\n")
                file.write(str(d))
                file.write("\n")
                for i in range(len(amplitude)):
                    file.write(f"{amplitude[i]},      {phase[i]}\n")
            
            # Plot amplitude vs. frequency
            plt.subplot(2, 1, 1)
            plt.stem(frequency_indices, amplitude, use_line_collection=(True))
            plt.title('Amplitude vs. Frequency')
            plt.xlabel('Frequency Index')
            plt.ylabel('Amplitude')
            
            # Plot phase shift vs. frequency
            plt.subplot(2, 1, 2)
            plt.stem(frequency_indices, phase, use_line_collection=(True))
            plt.title('Phase Shift vs. Frequency')
            plt.xlabel('Frequency Index')
            plt.ylabel('Phase Shift (radians)')
            
            plt.tight_layout()
            plt.show()
            
            amp, shift=read_file(output_file)
            ampExp, shiftExp = read_file("Output_Signal_DFT_A,Phase.txt")
            
            print(amp)
            print(ampExp)
            
            print(SignalComapreAmplitude(ampExp, amp))
            print(SignalComaprePhaseShift(shiftExp, shift))
            
        
    def calculate_idft():
            index = index_value.get()
            new_amplitude = amp_value.get()
            new_phase = shift_value.get()
        
            if new_amplitude != 0.0 or new_phase != 0.0:
                # Modify the DFT component at the specified index
                file_path = "frequency_components.txt"
                with open(file_path, 'r+') as file:
                    lines = file.readlines()
                    if index < len(lines):
                        components = lines[index].split(',')
                        if len(components) == 2:
                            if new_amplitude != 0.0:
                                components[0] = str(new_amplitude)
                            if new_phase != 0.0:
                                components[1] = str(new_phase)
                            lines[index] = ','.join(components) + '\n'
                        file.seek(0)
                        file.writelines(lines)
                        file.truncate()
            else:
                file_path = "Input_Signal_IDFT_A,Phase.txt"
        
            r = read_file2(file_path)
        
            d = len(r)
            X = np.zeros(d, dtype=complex)
            indices=[]
        
            for n in range(d):
                for k in range(d):
                    X[n] += (1 / d) * r[k] * np.exp(2j * math.pi * k * n / d)
            
            indices = [i for i in range(8)]
            
            # Plot amplitude vs. time
            plt.subplot(2, 1, 1)
            plt.plot(indices, X.real)
            plt.title('Amplitude vs. Time')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            
            plt.tight_layout()
            plt.show()
        
    window = tk.Tk()
    window.title("DFT Analyzer")
    window.geometry("300x400")
    window.configure(bg="MistyRose2")
        
    #Enter Sampling frequency
    Fs_label= tk.Label(window, text="Enter Sample Frequency :")
    Fs_label.place(x=80,y=30)
        
    Fs_entry = tk.Entry(window)
    Fs_entry.place(x=80,y=70)
        
    button = tk.Button(window, text="Calculate DFT", command=calculate_dft).place(x=100,y=100)
        
        #change amplitude and shift
    index_label= tk.Label(window, text="Enter index to change :")
    index_label.place(x=80,y=130)
    index_value = tk.IntVar()
    index_entry = tk.Entry(window, textvariable=index_value).place(x=80,y=160)
        
    amp_label= tk.Label(window, text="Enter new Amplitude value :")
    amp_label.place(x=80,y=200)
    amp_value = tk.IntVar()
    amp_entry = tk.Entry(window, textvariable=amp_value).place(x=80,y=230)
            
    shift_label= tk.Label(window, text="Enter new phase shift value :")
    shift_label.place(x=80,y=260)
    shift_value = tk.IntVar()
    shift_entry = tk.Entry(window, textvariable=shift_value).place(x=80,y=300)
        
    button2 = tk.Button(window, text="Calculate IDFT", command=calculate_idft).place(x=100,y=330)
        
    window.mainloop()

#---------fifth task---------
def fifth():
    def save_to_file( data, num_lines):
        file_path="save.txt"
        with open(file_path, 'w') as file:
            for i, value in enumerate(data[:num_lines], start=1):
                file.write(f"{i}\t{value}\n")

    def on_submit():
        
        DCT="DCT_input.txt"
        Rem_DC="DC_component_input.txt"              
        expecte_Dct="DCT_output.txt"
        expect_remove="DC_component_output.txt"
        
        Dct_result=Computing_DCT(DCT) 
        Remove_result=Remove_DC(Rem_DC)
        
        SignalSamplesAreEqual(expecte_Dct,Dct_result) 
        SignalSamplesAreEqual(expect_remove,Remove_result) 
        
        input_list = Dct_result
        
        print(input_list)
        num_lines = int(lines_entry.get())
        save_to_file( input_list, num_lines)
        result_label.config(text=f"{num_lines} lines have been saved to the file.")
        
    def Computing_DCT(file_path):
        indices,samples=read_file(file_path)
    
        res=[]
        N=len(samples)
        K=N
        for k in range(K):
            yk=0
            for n in range(N):
                yk += samples[n] * np.cos((np.pi/(4*N)) * (2*n-1) * (2*k-1))
            res.append(np.sqrt(2/N)*yk)
        
        plt.plot(indices, samples, label='Original Signal')
        plt.plot(indices[:len(res)], res, label='DCT Signal')
        plt.title('Original and DCT Signals')
        plt.legend()
        plt.show()
        return res    
        
    def Remove_DC(file_path):
        
        indices, samples=read_file(file_path)
        avg=np.mean(samples)
        new_val=samples-avg
        
        plt.plot(indices, samples, label='Original Signal')
        plt.plot(indices[:len(new_val)], new_val, label='Removed DC Signal')
        plt.title('Original and Removed DC Signals')
        plt.legend()
        plt.show()
        return new_val
    
    # Create the main window
    window = tk.Tk()
    window.title("Save List to File")
    window.configure(bg="MistyRose2")
    
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

#---------sixth task---------
def sixth():

    def smoothing():
        index,samples=read_file(input2_smoothing)
        WSize=int(windowSize_entry.get())
        result=[]
        for i in range(len(samples)):
            s=0
            if i+WSize>len(samples):break
            for j in range(i, i + WSize):
                s+=samples[j]
            result.append(s/WSize)   
            
        SignalSamplesAreEqual(output2_smoothing,result)
        plt.plot(index, samples, label='Original Signal')
        plt.plot(index[:len(result)], result, label=f'Smoothed Signal (Window Size = {WSize})')
        plt.title('Original and Smoothed Signals')
        plt.legend()
        plt.show()
        return  result  

        
    def idft(samples):
        file_path ="DC_component_input.txt"
        index,sampless=read_file(file_path) 
        d=len(samples)
        X=np.zeros(d, dtype=complex)
        for n in range(d):
            for k in range(d):
                X[n]+=samples[k]*np.exp(2j*math.pi*k*n/d)  
            X[n]*=(1/d)          
        
        plt.plot(index, sampless, label='Original Signal')
        plt.plot(index[:len(X)], X, label='Removed DC Component')
        plt.title('Original and Removed Dc Signals')
        plt.legend()
        plt.show()
        return X

    def calculate_dft():
        file_path ="DC_component_input.txt"
        index,samples=read_file(file_path) 
        d=len(samples)
        X=np.zeros(d, dtype=complex)
        for k in range(d):
            for n in range(d):
                X[k]+=samples[n]*np.exp(-2j*math.pi*k*n/d)     
        X[0]=0
        return idft(X)
                
        
    #smoothing
    input1_smoothing="Signal1.txt"
    input2_smoothing="Signal2.txt"
    output1_smoothing="MovAvgTest1.txt"
    output2_smoothing="MovAvgTest2.txt"

    def der():
        DerivativeSignal()

    def shift():
        indices, samples = read_file("Input Shifting.txt")
        
        const = float(constant_entry.get())
        result_signal = [index - const for index in indices]
        
        plt.plot(indices, samples, label='Original Signal')
        plt.plot(result_signal[:len(samples)], samples, label='Shifted Signal')
        plt.title('Original and Shifted Signals')
        plt.legend()
        plt.show()

    def fold():
        indices,samples=read_file("input_fold.txt")
        
        result_signal = [s for s in reversed(samples)]
        
        plt.plot(indices, samples, label='Original Signal')
        plt.plot(indices[:len(result_signal)], result_signal, label='Folded Signal')
        plt.title('Original and Folded Signals')
        plt.legend()
        plt.show()
        return indices, result_signal
        
    def fold_shift():
        indices_original, samples_original=read_file("input_fold.txt")
        indices,samples=fold()
        
        const = int(constant_entry.get())
        result_signal = [index + const for index in indices]
        
        Shift_Fold_Signal("Output_ShifFoldedby500.txt", result_signal, samples)
        # Shift_Fold_Signal("Output_ShiftFoldedby-500.txt", result_signal, samples)
        
        plt.plot(indices_original, samples_original, label='Original Signal')
        plt.plot(result_signal[:len(samples)], samples, label=f'Folded-Shifted Signal')
        plt.title('Original and Folded Signals')
        plt.legend()
        plt.show()

    window = tk.Tk()
    window.geometry("300x350")
    window.title("Task 6")
    window.configure(bg="MistyRose2")

    windowSize_label = tk.Label(window, text="Enter the window size:")
    windowSize_label.place(x=90, y=20)

    windowSize_entry = tk.Entry(window)
    windowSize_entry.place(x=90, y=50)

    smooth_button2 = tk.Button(window, text="Smooth", command=smoothing)
    smooth_button2.place(x=120, y=75)

    constant_label = tk.Label(window, text="Enter the constant:").place(x=100, y=110)
    constant_entry = tk.Entry(window)
    constant_entry.place(x=90, y=140)


    save_button2 = tk.Button(window, text="derivative", command=der).place(x=120, y=210)

    save_button3 = tk.Button(window, text="shift", command=shift).place(x=135, y=170)

    fold_button = tk.Button(window, text="fold", command=fold).place(x=136, y=245)

    foldShift_button = tk.Button(window, text="fold and shift", command=fold_shift).place(x=114, y=280)

    save_button2 = tk.Button(window, text="Remove DC", command=calculate_dft).place(x=120, y=315)

    window.mainloop()

#---------seventh task--------
def seventh():

    
    def expected_indices(indices1,indices2):
        indices=[]
        start=int(indices1[0])+int(indices2[0])
        end=int(indices1[len(indices1)-1])+int(indices2[len(indices2)-1])
    
        for i in range(start, end + 1):
            indices.append(i)
        return indices    
            
    def conv(signal1,signal2,end):
        samples=[]
        for i in range(end):
            k=0;ans=0
            for j in range(i, -1, -1):
                a=0;b=0
                if(k<len(signal1)):b=int(signal1[k])
                if(j<len(signal2)):a=int(signal2[j])  
                ans+=a*b;k+=1
            samples.append(ans)
                
        return samples


    def task_7():
        indices1,samples1=read_file('Input_conv_Sig1.txt')
        indices2,samples2=read_file('Input_conv_Sig2.txt')
        indices=expected_indices(indices1,indices2)
        samples=conv(samples1,samples2,len(indices))
        #print(indices,samples)
        ConvTest(indices,samples)
        
        plt.plot(indices1, samples1, label='Original Signal 1')
        plt.plot(indices2, samples2, label='Original Signal 2')
        plt.plot(indices[:len(samples)], samples, label='Convoluted Signal')
        plt.title('Original and Convoluted Signals')
        plt.legend()
        plt.show()
        
    
    
    root = tk.Tk()
    root.configure(bg="MistyRose2")
    root.title("Task 7 Executor")
    
    def execute_task_7():
        task_7()
        
    execute_button = tk.Button(root, text="convolution", command=execute_task_7)
    execute_button.pack(pady=20)
    root.mainloop()

#---------eighth task--------
def eighth():
        
    def corr():
        indices1, samples1= read_file("Corr_input signal1.txt")
        indices2, samples2= read_file("Corr_input signal2.txt")
        
        r12=[]
        x=np.zeros(len(samples1))
        d=len(samples2)
        for i in range(len(samples1)):
            for j in range(len(samples2)):
                up=np.roll(samples2,-i)
                # print(up)
                x[i]+=samples1[j]*up[j]/d
        return x,samples1,samples2,indices1, indices2

    def norm():
        x,s1,s2,i1,i2=corr()
        sum1=0
        sum2=0
        d=len(s2)
        normalized=[]
        for i in range(len(s1)):
            sum1+=s1[i]**2
        for j in range(len(s2)):
                sum2+=s2[j]**2
        for i in range(len(x)):
            normalized.append(x[i]/(np.sqrt(sum1*sum2)/d))
        print(normalized)
        
        plt.plot(i1, s1, label='Original Signal 1')
        plt.plot(i2, s2, label='Original Signal 2')
        plt.plot(i1[:len(normalized)], normalized, label='Correlated Signal')
        plt.title('Original and Correlated Signals')
        plt.legend()
        plt.show()
        
        Compare_Signals("CorrOutput.txt", i1, normalized)

    root = tk.Tk()
    root.configure(bg="MistyRose2")
    root.title("Task 8 Executor")
    execute_button = tk.Button(root, text="Correlation", command=norm)
    execute_button.pack(pady=20)
    root.mainloop()

#---------practical task------
def practical():   
        
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
        indices1,signal1=read_file("Input_conv_Sig1.txt")
        indices2,signal2=read_file("Input_conv_Sig2.txt")
        expected_iduces=expected_indices(indices1,indices2)
      
        sample1,sample2=append_zeros(signal1,signal2)

        result=sample1*sample2
        result=idft(result)
        ConvTest(expected_iduces,result)
        plt.plot(expected_iduces, result)
        plt.title('Convolution')
        plt.show()
       
        plt.plot(indices1, signal1, label='Original Signal 1')
        plt.plot(indices2, signal2, label='Original Signal 2')
        plt.plot(expected_iduces[:len(result)], result, label='Convoluted Signal')
        plt.title('Original and Convoluted Signals')
        plt.legend()
        plt.show()


    def fast_corr():
        indices1,signal1=read_file("Corr_input signal1.txt")
        indices2,signal2=read_file("Corr_input signal2.txt")
        
        expected_indices=[]
        for i in range(len(indices1)):expected_indices.append(i)
        sample1=calculate_dft(signal1)
        sample2=calculate_dft(signal2)

        for i in range(len(sample1)):
            sample1[i]=sample1[i].real+-1j*sample1[i].imag
     
        result=sample1*sample2
        result=1/(len(sample1))*idft(result)

        expected_output="Corr_Output.txt"
        plt.plot(indices1, signal1, label='Original Signal 1')
        plt.plot(indices2, signal2, label='Original Signal 2')
        plt.plot(expected_indices[:len(result)], result, label='Correlated Signal')
        plt.title('Original and Corrleated Signals')
        plt.legend()
        plt.show()
        
        Compare_Signals(expected_output,expected_indices,result)

    win = tk.Tk()
    win.configure(bg="MistyRose2")
    win.title("Practical Task Executer")

    # Correlation Button
    corr_button = ttk.Button(win, text="Correlation", command=fast_corr)
    corr_button.grid(pady=10, padx=20)

    # Convolution Button
    conv_button = ttk.Button(win, text="Convolution", command=fast_conv)
    conv_button.grid(pady=10, padx=20)

    # Add more widgets as needed

    root.mainloop()
#-----------main----------

# def execute_task(task_func):
#     task_func()

def display_task():
    selected_task = task_listbox.get(tk.ACTIVE)
    if selected_task in task_functions:
        task_functions[selected_task]()

root = tk.Tk()
root.geometry("400x500")
root.title("Task Selection")
root.configure(bg="MistyRose2")

# Creating a list of tasks
tasks = ["Task 1.1", "Task 1.2", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6", "Task 7", "Task 8", "Practical"]
    
# Creating a label
label = tk.Label(root, text="Select a task", font=('Arial', 20), bg="MistyRose2")
label.place(x=120, y=40)

# Creating a listbox
task_listbox = tk.Listbox(root, width=50)
task_listbox.place(x=45, y=100)

# Populating the listbox with tasks
for task in tasks:
    task_listbox.insert(tk.END, task)

# Creating a button to display the selected task
select_button = tk.Button(root, text="Select", command=display_task, width=20, height=2, background="#FFFFFF")
select_button.place(x=120, y=300)

# Mapping tasks to their corresponding functions
task_functions = {
"Task 1.1": first1,
"Task 1.2": first2,
"Task 2": second,
"Task 3": third,
"Task 4": fourth,
"Task 5":fifth,
"Task 6":sixth,
"Task 7":seventh,
"Task 8":eighth,
"Practical":practical
}

root.mainloop()