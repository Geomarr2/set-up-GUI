# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 09:12:13 2018

@author: User
"""

import numpy as np
import socket
import matplotlib.pyplot as plt
import csv
import AcqData
import logging

path = "C:/Users/User/Desktop/ModenVerwirbelungskammer/TestIIQdatan/"

bandwidth = 56e6 #Hz

AcqBW = 40e6
bandwidth = AcqBW #Hz
StartFreq = 1000e6
StopFreq = 1200e6
Z0 = 119.9169832 * np.pi  # Impedance of freespace
G_LNA= 10 #dB gain of the LNA
Lcable = -1  #dB cable losses
antennaEfficiency = 0.75 
r = 1.0  # Distance DUT to receiving antenna
color = ['r','b','m','c','g','y','hotpink','olive','coral','darkkhaki','orchid','lightblue','navy','rosybrown','cornflowerblue','lavenderblush','cadetblue','hotpink','olive','coral','darkkhaki','orchid','lightblue','navy','aliceblue']
path = "D:/Geomarr/Spectrum"
resolution = 100/107
bandwidth = 56e6 #Hz

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
    data =np.array(data, dtype=float)
    return data
    
def readIQDataBin(path,filename):
    arraydata =np.load(path + filename)
    return arraydata.T

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
    print()
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
    calfactor = 1000/50/523392/2                    
    return 10*np.log10(x*x*calfactor)
    
def trim_spectrum(spectrum):
    final_sample= 373852
    specsize=len(spectrum[0])
    AcqData._log.info("Usable number of Samples: %i "%(final_sample))
    AcqData._log.info("Spec length: %i "%(specsize))
    starttrim = int((specsize-final_sample)/2) 
    stoptrim = int(specsize-(specsize-final_sample)/2)
    freq = np.array(spectrum[0])
    fft= np.array(spectrum[1])
    return freq[starttrim:stoptrim],fft[starttrim:stoptrim]
            
def plot_stiched_spectrum(spectrum, color, resfact = 1):
    
   # trimspec = np.array(change_freq_channel(trim_spectrum(spectrum),resfact))
   # trimspec = np.array(trim_spectrum(spectrum))
    
    spec = to_decibels(spectrum[1])
    resolution =  0.1069943751528491*resfact
     
    #_log.info("Generating plots (%.2f to %.2f MHz)"%(lower/1e6,upper/1e6))
    plt.plot(spectrum[0]/1e6,spectrum[1], c=color)
    #plt.ylim(-80,0)

    
def applyWindow(complexIQdata):
    window = np.blackman(len(complexIQdata))
    return complexIQdata*window

def calCompFFT(complexIQdata):
    tempSpec = np.fft.fft(complexIQdata)
    complexSpec = np.fft.fftshift(tempSpec)
    # Create the frequency scale
    freqscale = creatFreqScale(centerFreq, bandwidth, len(complexIQdata))
    return freqscale, complexSpec

def convertFFT2powerSpectrum(spectrum):
    x = np.abs(spectrum[1])
    calfactor = 1000/50/len(spectrum[1])/2
                          
    return spectrum[0], 10*np.log10(x*x*calfactor)  

def calCCF(spectrum, CCF, r, Lcable, G_LNA, antennaEfficiency): # returns in [dBuV/m]
    spectrum[1]

    temp = -G_LNA - Lcable - (10.0 * np.log10(antennaEfficiency)) - CCF + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r *r)))) + 90.0
      
    return spectrum[0], spectrum[1]+temp



if __name__ == '__main__':
    CCFdata = readIQDatafileCCF(PathCCF,FileCCF) 
    Freqlist =generate_CFlist(StartFreq,StopFreq)
    tempCCFData = []
    tempSpec = []
    cfreq = 100e6
    filename="/IQWerte_100.0MHz.npy"
# Average CCF for each given centre frequency and AcqBW
    for j in range(len(CCFdata)):
        if cfreq == CCFdata[j][0] :
             CCF = CCFdata[j][1] # Chamber Calibration Factor in dBm
                      
     # calculate  calibrated spectrum and plot
    CCF = -20
    ReadFile = readIQDataBin(path,filename)
    iqData = convert2Complex(ReadFile)
    speccomp = calCompFFT(applyWindow(iqData))
    spec = convertFFT2powerSpectrum(speccomp)
    Spec = calCCF(spec, CCF, r, Lcable, G_LNA, antennaEfficiency)
    plot_stiched_spectrum(Spec,color[1])
        #plt.plot(tempSpec[i,0,:]/1e6,tempSpec[i,1,:])
    #print(CCF.shape)
   # spec = calCCF(spec, CCF[], r, Lcable, G_LNA, antennaEfficiency)
    #print(len(spec))♫
    #plt.plot(creatFreqScale(centerFreq,bandwidth,len(r)),convertFFT2powerSpectrum(spec))
    #plt.plot(spec[0],convertFFT2powerSpectrum(spec[1]))
       # plt.plot(Spec[0]/1e6,Spec[1])
    plt.ylabel("Power (dB)")
    plt.xlabel("Frequency (MHz) (resolution %.3f kHz)"%resolution)
    plt.show()
   