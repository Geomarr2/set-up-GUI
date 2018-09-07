# -*- coding: utf-8 -*-
"""
Created on Tue May 08 10:31:55 2018

@author: geomarr
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:01:55 2018

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 09:12:13 2018

@author: User
"""
#import os
import ctypes as c
import numpy as np
#import math as math
import matplotlib.pyplot as plt
#import time
#import logging
from threading import Timer,Event,Lock
#import spectra
import scipy.io
import csv

FileCCF="CCF.csv"
filename="RFISpectrum"
PathCCF = "D:/Geomarr/SNR/CCF/"
c = 299792458
AcqBW = 40e6
k = 1.38064852e-23 # Boltzmann constant
StartFreq = 1000e6
StopFreq = 1040e6
Z0 = 377#119.9169832 * np.pi  # Impedance of freespace
G_rx = 5 #dBi gain of the LPDA
G_LNA= 20 #dB gain of the LNA
Lcable = -1  #dB cable losses
antennaEfficiency = 0.75 
RBW = 107 # resolution bandwidth
r = 1.0  # Distance DUT to receiving antenna
path = "D:/Geomarr/compare spectrum/"

resolution = 1
integrationTime = 1.0          #integration time in sec
displayResolution = 1            # times 107Hz
usecase = 0  # 0 = plain data taking; 1 = calibrate data; 2 = acquire calibration data; 3 = start RFI data; 4 = acquire background data
GPU_integration_time = 2.0
color = ['y','hotpink','olive','coral','darkkhaki','orchid','lightblue','navy','rosybrown','cornflowerblue','lavenderblush','cadetblue','hotpink','olive','coral','darkkhaki','orchid','lightblue','navy','aliceblue','r','b','m','c','g']

def load_spectrum(path, filename):
    return np.load(path+filename)

    
def readIQDatafile(path, filename):
    data = []
    filedata = open(path+filename, "r")
    csvdata = csv.reader(filedata)
    for row in csvdata:
        data.append(row)
    filedata.close()
#    data =np.array(data, dtype=float)
    return data


def readSpectralData(path,filename):
    arraydata = np.load(path + filename)
    return arraydata
    

def convert2Complex(iqdata):
    return [(x + 1j*y) for x, y in iqdata]

def creatFreqScale(centerFreq, bandwidth, sample):
    freqList = []
    bandstart = centerFreq-bandwidth/2
    freqResolution = bandwidth/sample
    for i in range(sample):
        freqList.append(float((freqResolution*i+bandstart)))
    return np.array(freqList, dtype=float)

# genertes a list of standart center frequencies        
def generate_CFlist(startfreq, stopfreq):
    lowerlimit = 80e6 #  Hz Defines the lower frequency limit
    upperlimit = 6000e6 # Hz Defines the upper freqzency limit
    if startfreq > stopfreq: 
        temp = startfreq
        startfreq = stopfreq
        stopfreq = temp
    if lowerlimit >= startfreq and startfreq <= upperlimit: startfreq = lowerlimit
    if lowerlimit >= stopfreq and stopfreq <= upperlimit: stopfreq = upperlimit
    if startfreq > stopfreq: 
        temp = startfreq
        startfreq = stopfreq
        stopfreq = temp
        
    cfreq_tabelle = list(range(int(round((stopfreq/1e6-80)/40+0.499999))-int((startfreq/1e6-80)/40)))
    for i in range(len(cfreq_tabelle)):
        cfreq_tabelle[i]= ((i+int((startfreq/1e6-80)/40))*40+100)*1e6
    return cfreq_tabelle
    
def change_freq_channel(spectrum, factor):
    outputChannel = int(len(spectrum[0])/factor)
    outputfreqlist = np.zeros(outputChannel)
    outputspeclist = np.zeros(outputChannel)
    for i in range(outputChannel):
        outputfreqlist[i] = np.mean(spectrum[0][i*factor:(i*factor+factor)])
        outputspeclist[i] = np.mean(spectrum[1][i*factor:(i*factor+factor)])
    return outputfreqlist, outputspeclist
    
def to_decibels(x): 
    x = x*(1/(131072)) # for an 40 MHz Acq BW IQ block naxsize 131 072 samples
    calfactor = 1000/50                  
    return 10*np.log10(x*x*calfactor)
    
def trim_spectrum(spectrum):
    final_sample= 373852
    specsize=len(spectrum[0][:])
    starttrim = int((specsize-final_sample)/2) 
    stoptrim = int(specsize-(specsize-final_sample)/2)
    freq = np.array(spectrum[0])
    fft= np.array(spectrum[1])
    return freq[starttrim:stoptrim],fft[starttrim:stoptrim]
    
def plot_stiched_spectrum(spectrum, color, resfact = 1, yaxis = "Power [dBm]", title = "Power spectrum", label='none'):
    
    #trimspec = np.array(change_freq_channel(trim_spectrum(spectrum),resfact))
    #trimspec = np.array(trim_spectrum(spectrum))
    
    #spec = to_decibels(trimspec[1])
    resolution =  0.1069943751528491*resfact
    #_log.info("Generating plots (%.2f to %.2f MHz)"%(lower/1e6,upper/1e6))
    plot = plt.plot(spectrum[0]/1e6,spectrum[1], c=color, label=label)
    #plt.ylim(-80,0)
    plt.title(title)
    plt.ylabel(yaxis)
    plt.xlabel("Frequency (MHz) (resolution %.3f kHz)"%resolution)
    return plot

def freq_scale(true_bw,buffer_size,cfreq):
    bw = true_bw
    resolution = bw/buffer_size
    lower = cfreq-bw/2
    upper = cfreq+bw/2
    return np.linspace(lower,upper,buffer_size)
    
def spectra_linear(true_bw,buffer_size,cfreq,mean_spectra): 
    #max_specs = to_decibels(np.fft.fftshift(np.array(fft.max_spectra)))
    
    return freq_scale(true_bw,buffer_size,cfreq), np.fft.fftshift(np.array(mean_batches(mean_spectra)))
    
def applyWindow(complexIQdata, centerFreq):
    window = np.blackman(len(complexIQdata))
    return complexIQdata*window

def calCompFFT(complexIQdata):
    tempSpec = np.fft.fft(complexIQdata)
    complexSpec = np.fft.fftshift(tempSpec)
    # Create the frequency scale
    freqscale = creatFreqScale(centerFreq, AcqBW, len(complexIQdata))
    return freqscale, complexSpec

def convertFFT2powerSpectrum(spectrum):
    x = np.abs(spectrum)
    calfactor = 1000/50/len(spectrum)/2     
    return 10*np.log10(x*x*calfactor)  

def plot_stiched_spectrum_old(spectrum, color, resfact = 1, yaxis = "Power [dBm]", title = "Power spectrum", label='none'):
    trimspec = np.array(change_freq_channel(trim_spectrum(spectrum),resfact))
    #trimspec = np.array(trim_spectrum(spectrum))
    
    spec = to_decibels(trimspec[1])
    resolution =  0.1069943751528491*resfact
    
    #_log.info("Generating plots (%.2f to %.2f MHz)"%(lower/1e6,upper/1e6))
    plot = plt.plot(trimspec[0]/1e6,spec,c=color, label=label)
    plt.title(title)
    plt.ylabel(yaxis)
    plt.xlabel("Frequency (MHz) (resolution %.3f kHz)"%resolution)
    return plot
if __name__ == '__main__':
    CCFdata = readIQDatafileCCF(PathCCF,FileCCF) 
    sample = 523852 
    BW = AcqBW
    cfreq = generate_CFlist(StartFreq,StopFreq)
    cfreq = cfreq[0]
    sampling_rate = 40e6
    
    tempSpec = []
    CCF = []
    ReadFile = []
    #Spectrum_dBuVperM = 0
    Prx_dB = []
    Prfi = []
    Tsys = []
# obtain power in dBm from IQ voltageacquired through tektronix GUI
    Icomp = []
    Qcomp = [] 
    trimspec = []

    NF_rsa_dB = 0
    integrationTime = 1
    # set the CCF freq range
    upperfreq = cfreq + BW/2 
    lowerfreq = cfreq - BW/2
    for i in range((int(lowerfreq/1e6)-100), (int(upperfreq/1e6)-100),1):
            CCF.append(CCFdata[i])
    CCF = np.array(CCF, dtype = float) 
    
    # Load data Power Spectrum in dBm
    matPowerSpec = readIQDatafile(path,'1000_1040MHz_dBm.csv')
    
    # change freq spectrum could rescale the fft depending on what display resolution is prefered  
    trim = matPowerSpec[128:800][:] 
    FFT_size = len(trim)
    for x in range(len(trim)):
        trimspec.append(float(trim[x][0]))
    trimspe = np.array(trimspec, dtype = float)
    f = np.array(freq_scale(sampling_rate,FFT_size,cfreq), dtype=float)

    Spec_GUI = f, trimspe
    # plot raw data
    plt.figure()
    title = "Power spectrum"
    yaxis = "Power  [dBm]"
    Tekgui, = plot_stiched_spectrum(Spec_GUI, "b",displayResolution, yaxis, title, "Teketronix GUI")
    # Load data IQ data
    spectrum = load_spectrum('D:/Geomarr/compare spectrum/','RFISpectrum1020MHz.npy')

    python, = plot_stiched_spectrum_old(spectrum, "k",displayResolution, yaxis, title, "Calculated using python FFT")
    plt.legend(handles = [Tekgui, python])
    plt.grid()
    plt.show()
   