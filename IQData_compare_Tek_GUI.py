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
path = "D:/Geomarr/tek signalVu/"

resolution = 1
integrationTime = 1.0          #integration time in sec
displayResolution = 1            # times 107Hz
usecase = 0  # 0 = plain data taking; 1 = calibrate data; 2 = acquire calibration data; 3 = start RFI data; 4 = acquire background data
GPU_integration_time = 2.0
color = ['y','hotpink','olive','coral','darkkhaki','orchid','lightblue','navy','rosybrown','cornflowerblue','lavenderblush','cadetblue','hotpink','olive','coral','darkkhaki','orchid','lightblue','navy','aliceblue','r','b','m','c','g']

def load_spectrum(path, filename):
    return np.load(path+filename)

def readIQDatafileCCF(path, filename):
    data = []
    with open(path+filename, "r") as filedata:
        csvdata = csv.reader(filedata, delimiter = ',')
        for row in csvdata:
            data.append(row)
        filedata.close()
        data =np.array(data, dtype = 'float')    
    return data
    
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
    
def to_decibels(x): #dBm
    calfactor = 1000/50/len(x)/2                   
    return 10*np.log10(x*calfactor)
    
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

def calCCF(s, CCF, r, Lcable, G_LNA, antennaEfficiency, BW, centreFreq): # returns in [dBuV/m]
    tempS = []
    for i,k in enumerate(CCF[:,0]):
        for n,x in enumerate(s[0]):
            if x > k and x < (k+1e6): 
                tempSpec =  -G_LNA + Lcable - (10.0 * np.log10(antennaEfficiency)) - CCF[i,1] + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r * r)))) + 90.0
                tempS.append(tempSpec)
    tempS = tuple(tempS)
    return s[0],s[1] + tempS[1]
    
def Temp_Sys(NF_rsa, Lcable, G_LNA, RBW): # returns in [dBuV/m]
    NF_dB = NF_rsa + Lcable - G_LNA
    Tsys = pow(10, (NF_dB/10))/(k*RBW)
    return Tsys

def Sigma(TSys, L, N):
    # L is the length of the FFT (samples/sec)
    # N number of FFT being average (the size)
    Sigma = (k*TSys*AcqBW)/(L*np.sqrt(N))
    return Sigma 

def RFI_to_Simga(E_rfi,sigma,G_rx,f): # returns in [dBuV/m]
    snr = E_rfi + G_rx - 20*np.log10(f) + 20*np.log10(c/(4*np.pi*r)) + 10*np.log10(1/sigma) - 30
                                     
    return snr
    
def mean_NoiseFloor(NF,upperfreq, lowerfreq):
    meanNF = 0
    count = 0
    trimspec = np.array(NF)
    print(len(trimspec[1,:]))
    for i,x in enumerate(trimspec[0,:]):
        if x >= lowerfreq and x <= upperfreq:
            meanNF = trimspec[1,i] + meanNF
            count = 1 + count
            
    return meanNF/count
    
def convert_EField_Intensity_dBuVPerM2Power(EfieldPerMeter, antennaEfficiency):
    EField_Volt = EfieldPerMeter - (10.0 * np.log10(antennaEfficiency)) # E-field antenna correction factor convert
    EField_Power_dBm = EField_Volt - 107 # convert volts to power dBm for a 50 Ohm system
    EField_Power_dB = EField_Power_dBm - 30 # dBm convert to dB
    return EField_Power_dB
    
def EField_rfi(E_rx,NF_rsa_dB):
    E_rfi = pow(10, E_rx/10) - pow(10, NF_rsa_dB/10)
    return 10*np.log10(E_rfi)
    
def Efield_rx(spec,G_rx, G_LNA, Lcable):# E(rfi+NF)
    E_rx = spec[1] + 20*np.log10(spec[0]) - G_rx - G_LNA + Lcable + (10.0 * np.log10((4*np.pi*Z0)/c*c)) + 90
                
    return E_rx
    
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
    

    # calculate the calibrated data using the GUI power spec
    #Spectrum_dBuVperM = calCCF(Spec, CCF, r, Lcable, G_LNA, antennaEfficiency, BW, cfreq) 
    #TekguiCAL, = plot_stiched_spectrum(Spectrum_dBuVperM, "b",displayResolution, yaxis, title, "Teketronix GUI calibrated data")
    # Load data IQ data
    matIQdata = scipy.io.loadmat('D:/Geomarr/tek signalVu/IQdata.mat')
    IQdata = matIQdata['Y']
    buffer_size = len(IQdata)
    spec = np.fft.fft(IQdata)
    freq = np.array(freq_scale(sampling_rate,buffer_size,cfreq), dtype=float)
    r = [x * 1 for x in abs(spec)**2]
    r = np.fft.fftshift(r)
    spectrum = freq,to_decibels(r)
    #trimspec = np.array(change_freq_channel(trim_spectrum(spectrum),1))
    python, = plot_stiched_spectrum(spectrum, "k",displayResolution, [], title, "Calculated using python FFT")
    
    # Load data Power Spectrum in dBm
    matPowerSpec = readIQDatafile(path,'testdataIQdatamore2.csv')
    
    # change freq spectrum could rescale the fft depending on what display resolution is prefered  
    trim = matPowerSpec[128:800][:] 
    FFT_size = len(trim)
    for x in range(len(trim)):
        trimspec.append(float(trim[x][0]))
    trimspe = np.array(trimspec, dtype = float)
    f = np.array(freq_scale(sampling_rate,FFT_size,cfreq), dtype=float)

    Spec = f, trimspe
    # plot raw data
    plt.figure(1)
    title = "Power spectrum"
    yaxis = "Power  [dBm]"
    Tekgui, = plot_stiched_spectrum(Spec, "b",displayResolution, yaxis, title, "Teketronix GUI")
    
    plt.legend(handles = [Tekgui, python])
    plt.grid()
    plt.show()
    

#    Icomp.append(trimspec[x][0]+trimspec[x][1]*i)
#    Qcomp.append(trimspec[x][2]+trimspec[x][3]*i)
#    Icomp = np.array(Icomp, dtype = float)
#    Qcomp = np.array(Qcomp, dtype = float)

#    Spectrum_dBuVperM = calCCF(tempSpec, CCF, r, Lcable, G_LNA, antennaEfficiency, BW, centerFreq)
    # plot calibrated data
#    plt.figure(3)
#    title = "Calibrated data"
#    yaxis = "Electric field strenght  [dBuV/m]"
#    plot_stiched_spectrum(Spectrum_dBuVperM, "hotpink", displayResolution, yaxis, title)
    

   #