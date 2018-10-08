#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:02:47 2017

@author: gwieching
"""
__author__ = 'gwieching'


import numpy as np
import socket
import matplotlib.pyplot as plt
import csv


path = "D:/Geomarr/Spectrum/"
#filename="sine40mVRMS.txt"
filename="RFISpectrum1020MHz.csv"
#filename="IQWerte_100.0MHz.txt"
#filename="IQWerte_100.0MHz.npy"
#filename="Spectrum_110.0MHz.npy"
centerFreq = 100e6 #Hz
bandwidth = 56e6 #Hz


def readIQDatafile(path, filename):
    data = []
    filedata = open(path+filename, "r")
    csvdata = csv.reader(filedata)
    print(csvdata)
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


def calCompFFT(complexIQdata):
    tempSpec = np.fft.fft(complexIQdata)
    complexSpec = np.fft.fftshift(tempSpec)
    # Create the frequency scale
    freqscale = creatFreqScale(centerFreq, bandwidth, len(complexIQdata))
    return freqscale, complexSpec

def applyWindow(complexIQdata):
    window = np.blackman(len(complexIQdata))
    return complexIQdata*window

def convertFFT2powerSpectrum(spectrum):
    x = np.abs(spectrum[1])
    calfactor = 1000/50/len(spectrum[1])/2
                          
    return spectrum[0], 10*np.log10(x*x*calfactor)

def convertPower2ElecFieldStrength(spectrum): # returns in [dBuV/m]
    Z = 119.9169832 * np.pi  # Impedance of freespace
    gLNA= 10.0 #dB
    gCable = -1.0  #dB
    antennaEfficiency = 0.75 
    r = 1.0  # Distance DUT to Antenna
    CCF =-20.0  # Chamber Calibration Factor in dBm
    spectrum[1]

    temp = -gLNA - gCable - (10.0 * np.log10(antennaEfficiency)) - CCF + (10.0 * np.log10(Z / (4.0 * np.pi * (r *r)))) + 90.0
      
    return spectrum[0], spectrum[1]+temp
            
#def removeNoiseFloor(spectrum, noisespectrum):
#    if np.array_equal(spectrum[0],noisespectrum[0]):
#        
#    
#    


if __name__ == '__main__':
    iqData = (readIQDatafile(path,filename))
    print(iqData)
    speccomp = calCompFFT(applyWindow(iqData))
    spec = convertFFT2powerSpectrum(speccomp)
    
    spec = convertPower2ElecFieldStrength(spec)
    
    #plt.plot(creatFreqScale(centerFreq,bandwidth,len(r)),convertFFT2powerSpectrum(spec))
    #plt.plot(spec[0],convertFFT2powerSpectrum(spec[1]))
    plt.plot(spec[0]/1e6,spec[1])
    #plt.plot(specw[0]/1e6,specw[1])
    
    
  
  
#    plt.plot(creatFreqScale(centerFreq,bandwidth,len(r)),convertFFT2powerSpectrum(specw))
  
    
    
    
    
    