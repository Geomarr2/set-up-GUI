# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 16:42:11 2018

@author: User
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
import AcqData

pathCaldata = "D:/RFIData/Effelsberg/Webcam-DCS7513/20171128/01_KalibrationData_background/" #Path to save the calibration data spectra
DataPath = "D:/Geomarr/Spectrum/"       #Path to save the spectra
#integrationTime = 1           #integration time in sec
startFreqMHz = 1000                      #start frequency in MHz
stopFreqMHz  = 1040            #stop frequency in MHz
displayResolution = 10             # times 107Hz
title = "Power Spetrum"
usecase = 4  # 0 = plain data taking; 1 = calibrate data; 2 = acquire calibration data; 3 = start RFI data; 4 = acquire background data
GPU_integration_time = 3
color = "c"
i = 0
min_freq = startFreqMHz*1e6
max_freq = stopFreqMHz*1e6
integration_time = 3


 
if __name__ == "__main__":
 #   _log.setLevel(logging.INFO) 
    #main(300e6,320e6,2.00,path)
    
    for i in range(0, 1):
        DataPath = DataPath + str(i)        
        AcqData._log.setLevel(logging.INFO)
        device = AcqData.TektronixDevice()
        batch = int(GPU_integration_time*device.bw/device.buffer_size)
        dump_time = batch*device.buffer_size/device.bw
        AcqData._log.info("Setting dump time to %.3f seconds."%(dump_time))
        fft = AcqData.spectra.FftDetect(device.buffer_size,batch)
        device.run()
        Cfreqlist = AcqData.generate_CFlist(min_freq,max_freq)
        act_steps =1
        print("Cfreqlist ="+str(Cfreqlist))
        for i in range(len(Cfreqlist)):
            device.set_cfreq(int(Cfreqlist[i]))
            AcqData._log.info("Frequency step %i of %i."%(act_steps,len(Cfreqlist)))
            AcqData.process_stream(device,fft,integration_time)
            spectrum = AcqData.spectra_linear(device,fft)
            AcqData._log.info("Frequency step %i of %i."%(act_steps,len(Cfreqlist)))
        
            AcqData.plot_stiched_spectrum(spectrum, color)
            AcqData.save_spectrum2file(spectrum, DataPath,"RFISpectrum"+str(int(device.cfreq/1e6))+ "MHz" )
        #write_spec2file(spectrum, DataPath,"RFISpectrum"+str(int(device.cfreq/1e6))+ "MHz" )
        
            AcqData._log.info("Size of fft mean_spectra: "+ str(np.size(fft)))
            fft.clear()
            act_steps +=1
    
        device.stop()
        plt.tight_layout()
        plt.title(title)
    
        plt.show()
        fft.destroy()
        device.disconnect()
        
        #AcqData.acquire_data(startFreqMHz*1e6,stopFreqMHz*1e6,integrationTime,DataPath,GPU_integration_time, pathCaldata, displayResolution, title, usecase, color)
