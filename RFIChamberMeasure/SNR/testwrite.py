# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:50:10 2017

@author: Geomarr
"""
import os
import ctypes as c
import numpy as np
import math as math
import matplotlib.pyplot as plt
import time
import logging
from threading import Timer,Event,Lock
import spectra
import tektronix_errors

DataPath = "D:/Geomarr/Spectrum/"       #Path to save the spectra 
integrationTime = 1           #integration time in sec
startFreqMHz = 1000                        #start frequency in MHz
stopFreqMHz  = 1080            #stop frequency in MHz
displayResolution = 10             # times 107Hz
title = "Power Spetrum"
usecase = 4  # 0 = plain data taking; 1 = calibrate data; 2 = acquire calibration data; 3 = start RFI data; 4 = acquire background data
GPU_integration_time = 3
color = "m"



def main(min_freq,max_freq,integration_time,DataPath,GPU_integration_time, displayResolution = 1, title = "", usecase = 0, color = "r"):
    # usecase:  0 = plain data taking
    #           1 = calibrate data
    #           2 = acquire calibration data
    #           3 = start RFI data
    #           4 = acquire background data
    
    device = TektronixDevice()
    batch = int(GPU_integration_time*device.bw/device.buffer_size)
    dump_time = batch*device.buffer_size/device.bw
    _log.info("Setting dump time to %.3f seconds."%(dump_time))
    fft = spectra.FftDetect(device.buffer_size,batch)
    """
    ################################################################
    Start IQ streaming 
    ################################################################
    """
    device.run()
    Cfreqlist = generate_CFlist(min_freq,max_freq)
    act_steps =1
    
    for i in range(len(Cfreqlist)):
        device.set_cfreq(int(Cfreqlist[i]))
        _log.info("Frequency step %i of %i."%(act_steps,len(Cfreqlist)))
        process_stream(device,fft,integration_time)
        spectrum = spectra_linear(device,fft)
        device.write_spec2file(int(Cfreqlist[i]))
        _log.info("Frequency step %i of %i."%(act_steps,len(Cfreqlist)))
        if usecase == 0:
            plot_stiched_spectrum(spectrum, color)
        if usecase == 1:
            save_spectrum2file(spectrum, DataPath,"RFISpectrum"+str(int(device.cfreq/1e6))+ "MHz" )
            plot_stiched_spectrum(calibrateData(spectrum, Cfreqlist[i], DataPath),color,displayResolution)
        if DataPath != "" and usecase == 2:
            plot_stiched_spectrum(spectrum, color)
            save_spectrum2file(generate_blanklist(spectrum), path,"BlanckFile"+str(int(device.cfreq/1e6))+ "MHz" )
            save_spectrum2file(spectrum,DataPath,"50OhmRefSpectrum"+str(int(device.cfreq/1e6))+ "MHz" )
        if DataPath != "" and usecase == 3:
            plot_stiched_spectrum(spectrum, color, displayResolution)
            save_spectrum2file(spectrum, DataPath,"RFISpectrum"+str(int(device.cfreq/1e6))+ "MHz" )
            # Load Background spectra and plot
            filename = "Background"+str(int(device.cfreq/1e6))+ "MHz.npy"
            plot_stiched_spectrum(load_spectrum(pathCaldata, filename), color,displayResolution)            
        if DataPath != "" and usecase == 4: #Generates and saves the background spectra
            plot_stiched_spectrum(spectrum, color,displayResolution)
            save_spectrum2file(spectrum, pathCaldata,"Background"+str(int(device.cfreq/1e6))+ "MHz" )
        _log.info("Size of fft mean_spectra: "+ str(np.size(fft)))
        fft.clear()
        act_steps +=1
    
    device.stop()
    plt.tight_layout()
    plt.title(title)
    
    plt.show()
    fft.destroy()
    device.disconnect()

if __name__ == "__main__":
    _log.setLevel(logging.INFO) 
    main(startFreqMHz*1e6,stopFreqMHz*1e6,integrationTime,DataPath,GPU_integration_time, displayResolution, title, usecase, color)