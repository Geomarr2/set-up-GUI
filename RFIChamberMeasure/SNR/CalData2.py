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

import numpy as np
import socket
import matplotlib.pyplot as plt
import csv
import AcqData
import logging


FileCCF="CCF4.csv"
filename="RFISpectrum"
PathCCF = "D:/Geomarr/Spectrum/"
#filename="IQWerte_100.0MHz.npy"
#filename="Spectrum_110.0MHz.npy"

AcqBW = 40e6
k = 1.38064852e-23 # Boltzmann constant
bandwidth = AcqBW # Hz
StartFreq = 1000e6
StopFreq = 1040e6
Z0 = 119.9169832 * np.pi  # Impedance of freespace

G_LNA= 20 #dB gain of the LNA
Lcable = -1  #dB cable losses
antennaEfficiency = 0.75 

r = 1.0  # Distance DUT to receiving antenna
path = "D:/Geomarr/Spectrum/ChangeIntegrationTime"+"_gLNA"+str(G_LNA)+"_Lcable"+str(Lcable)+"_EffAnt"+str(antennaEfficiency)+"/ "
resolution = 1
DataPath = "D:/Geomarr/Spectrum/FullSpectrum"       #Path to save the spectra
integrationTime = 1           #integration time in sec
displayResolution = 1            # times 107Hz
usecase = 0  # 0 = plain data taking; 1 = calibrate data; 2 = acquire calibration data; 3 = start RFI data; 4 = acquire background data
GPU_integration_time = 3
color = ['y','hotpink','olive','coral','darkkhaki','orchid','lightblue','navy','rosybrown','cornflowerblue','lavenderblush','cadetblue','hotpink','olive','coral','darkkhaki','orchid','lightblue','navy','aliceblue','r','b','m','c','g']

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
    specsize=len(spectrum[0][:])
    AcqData._log.info("Usable number of Samples: %i "%(final_sample))
    AcqData._log.info("Spec length: %i "%(specsize))
    starttrim = int((specsize-final_sample)/2) 
    stoptrim = int(specsize-(specsize-final_sample)/2)
    freq = np.array(spectrum[0])
    fft= np.array(spectrum[1])
    return freq[starttrim:stoptrim],fft[starttrim:stoptrim]
            
def plot_stiched_spectrum(spectrum, c, resfact = 1):
    trimspec = np.array(change_freq_channel(trim_spectrum(spectrum),resfact))
    #trimspec = np.array(trim_spectrum(spectrum))
    print(c)
    spec = to_decibels(trimspec[1][:])
    resolution =  0.1069943751528491*resfact
    #_log.info("Generating plots (%.2f to %.2f MHz)"%(lower/1e6,upper/1e6))
    plt.plot(trimspec[0][:]/1e6,spec, c)
    #plt.ylim(-80,0)
    plt.ylabel("Electric field strenght  [dBu/m]")
    plt.xlabel("Frequency (MHz) (resolution %.3f kHz)"%resolution)

def freq_scale(true_bw,buffer_size,cfreq):
    bw = true_bw
    resolution = bw/buffer_size
    lower = cfreq-bw/2
    upper = cfreq+bw/2
    return np.linspace(lower,upper,buffer_size)
    
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

def calCCF(spectrum, CCF, r, Lcable, G_LNA, antennaEfficiency, BW, centreFreq): # returns in [dBuV/m]
    spectrum[0,1] = 0 
    tempSpec = spectrum
    BW = BW*1e6
   # upperfreq = centreFreq + BW/2 
   #lowerfreq = centreFreq - BW/2 
   # CCFnew = CCF[(int(lowerfreq/1e6)-100):1:(int(upperfreq/1e6)-100)]
    for i in range(len(CCF[:,0])):
       for n,x in enumerate(spectrum[:,0]):
           #if x <= CCF[i,0] and x >= CCF[i,0]:
            tempSpec[n,1] =  -G_LNA - Lcable - (10.0 * np.log10(antennaEfficiency)) - CCF[i,1] + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r *r)))) + 90.0
               #tempSpec.append(temp)
    #tempSpec = np.array(tempSpec, dtype = 'float')            

    #        tempSpec[n] = np.nan
    #for i in range((int(lowerfreq/1e6)-100), (int(upperfreq/1e6)-100),1):
    #    for j in range (int(len(spectrum[:,0])/56 ) ):
    #        temp = -G_LNA - Lcable - (10.0 * np.log10(antennaEfficiency)) - CCF[i,1] + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r *r)))) + 90.0
    #        tempSpec.append(temp)
    #tempSpec = np.array(tempSpec, dtype = 'float')
    return spectrum[:,0],  spectrum[:,1] #+ tempSpec[:,1]
    
def Temp_Sys(NF_rsa, Lcable, G_LNA): # returns in [dBuV/m]
    NF_dB = NF_rsa + Lcable - G_LNA
    Tsys = (10^(NF_dB/10))/(k*AcqBW)
    return Tsys
    
def Simga(TSys, L, N):
    # L is the length of the FFT (samples/sec)
    # N number of FFT being average (the size)
    Sigma = (k*TSys*AcqBW)/(L*np.sqrt(N))
    return 10*np.log10(Sigma)

def RFI_to_Simga(spectrum, CCF, r, Lcable, G_LNA, antennaEfficiency, BW, centreFreq): # returns in [dBuV/m]
    spectrum[0,1] = 0 
    tempSpec = spectrum
    BW = BW*1e6
   # upperfreq = centreFreq + BW/2 
   #lowerfreq = centreFreq - BW/2 
   # CCFnew = CCF[(int(lowerfreq/1e6)-100):1:(int(upperfreq/1e6)-100)]
    for i in range(len(CCF[:,0])):
       for n,x in enumerate(spectrum[:,0]):
           #if x <= CCF[i,0] and x >= CCF[i,0]:
            tempSpec[n,1] =  -G_LNA - Lcable - (10.0 * np.log10(antennaEfficiency)) - CCF[i,1] + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r *r)))) + 90.0
               #tempSpec.append(temp)
    #tempSpec = np.array(tempSpec, dtype = 'float')            

    #        tempSpec[n] = np.nan
    #for i in range((int(lowerfreq/1e6)-100), (int(upperfreq/1e6)-100),1):
    #    for j in range (int(len(spectrum[:,0])/56 ) ):
    #        temp = -G_LNA - Lcable - (10.0 * np.log10(antennaEfficiency)) - CCF[i,1] + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r *r)))) + 90.0
    #        tempSpec.append(temp)
    #tempSpec = np.array(tempSpec, dtype = 'float')
    return spectrum[:,0],  spectrum[:,1] #+ tempSpec[:,1]
    
def mean_NoiseFloor(NF):
    meanNF =0
    for x in range(len(NF)):
        meanNF = NF[x] + meanNF
    return meanNF/len(NF)
    
if __name__ == '__main__':
    CCFdata = readIQDatafileCCF(PathCCF,FileCCF) 
    Freqlist =generate_CFlist(StartFreq,StopFreq)
    sample = 523852 
    BW = 40*1e6
    centreFreq = StartFreq+BW/2
    tempSpec = []
    CCF = []
    Length_freq = len(Freqlist) 
    # set the CCF freq range
    upperfreq = centreFreq + BW/2 
    lowerfreq = centreFreq - BW/2
    for i in range((int(lowerfreq/1e6)-100), (int(upperfreq/1e6)-100),1):
            CCF.append(CCFdata[i])
    CCF = np.array(CCF, dtype = 'float')        
    print()
    fileName = filename+str(int(Freqlist[0]/1e6))+"MHz_IntegrationTime_"+str(integrationTime)+".npy"
    ReadFile = readIQDataBin(path,fileName)
    Spec = calCCF(ReadFile, CCF, r, Lcable, G_LNA, antennaEfficiency, BW, Freqlist[0])
    plot_stiched_spectrum(Spec,color[0] )
        #plt.plot(tempSpec[i,0,:]/1e6,tempSpec[i,1,:])
    #print(CCF.shape)
   # spec = calCCF(spec, CCF[], r, Lcable, G_LNA, antennaEfficiency)
    #print(len(spec))♫
    #plt.plot(creatFreqScale(centerFreq,bandwidth,len(r)),convertFFT2powerSpectrum(spec))
    #plt.plot(spec[0],convertFFT2powerSpectrum(spec[1]))
       # plt.plot(Spec[0]/1e6,Spec[1])
    #plt.ylabel("Power (dB)")
    #plt.xlabel("Frequency (MHz) (resolution %.3f kHz)"%resolution)
    plt.show()
   