# Micah Clarke
# ID: 1001288866

import numpy as np
import scipy as sc
from scipy.signal import freqz
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import math


def lowPassFilter(signal,fs,m,l):
    
    #Cut-off frequency
    fc = 7500
    # Normalized frequency
    ft = fc/fs
    # Weighted values for Low Pass Filter
    h = []
    # Windowing values
    w = []
    
    # Producing the weight values to plot the frequency response with no hamming window from the filter coefficients
    i = 0
    while i < len(l):
        if l[i] == 50:
            h = np.append(h,2*ft)
        else:
            x = math.sin(2*np.pi*ft*(l[i]-m/2))
            y = np.pi*(l[i]-m/2)
            z = x/y
            h = np.append(h,z)
        i += 1

    # Producing the hamming coefficients to plot the frequenct response 
    i = 0
    while i < len(l):
        x1 = 0.54 - (0.46*math.cos((2*np.pi*l[i])/m))
        w = np.append(w,x1)
        i += 1

    # Element-wise multiplication between h[n] and w[n]
    c = np.multiply(h,w)

    # Convolving the signal with new filter coefficients
    responseValues = np.convolve(signal,c)

    
    plotAndSave(h,c,responseValues,fs)
   

    
def plotAndSave(h,c,responseValues,samplingRate):

    # Frequency response of lowpass filter w/out window
    x,y = freqz(h,1)
    # Frequency response of lowpass filter w/window
    x1,y1 = freqz(c,1)

    # Scale values to readable values for the media player to read and play
    scaled = np.int16(responseValues/np.max(np.abs(responseValues)) * 32767)
    write("cleanMusic.wav", samplingRate, scaled)

    
    # Original Signal Plot
    plt.plot(x,abs(y))

    # Window Signal Plot
    plt.plot(x1,abs(y1),'C1')
             
    plt.gca().legend(('original','windowed'))
    plt.title("Frequency Response")
    plt.show()
    


def main():
    # Import wav values and sampling rate
    fs, signal = sc.io.wavfile.read("P_9_2.wav")
    print(len(signal))
    # Filter length values list
    l = []
    l = np.arange(0,101,1)
    
    # Filter order
    m = len(l) - 1

    lowPassFilter(signal,fs,m,l)
        

main()


    
