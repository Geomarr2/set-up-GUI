# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:46:56 2018

@author: User
"""

import os
import ctypes
from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import time

# Import the RSA300API DLL
os.chdir("C:\\Tektronix\\RSA_API\\lib\\x64")
rsa306 = cdll.LoadLibrary("RSA_API.dll")

numFound = c_int()
intArray = c_int * 10
durationMsec = c_int(10000) 
bwHz_req = c_double(40e6)
bwHz_act = c_double(0)
sRate = c_double(0)
deviceIDs = intArray()
srSps = c_double(0)
temp = "Ein gaaaaaaanz langer String"
deviceSerial = c_char_p(temp.encode('utf-8'))
deviceType = c_char_p(temp.encode('utf-8'))
apiVersion = c_char_p(temp.encode('utf-8'))
enableSpectrum = c_bool()
enableSpectrogram = c_bool()

#get API version
#rsa306.DEVICE_GetAPIVersion(apiVersion)
#print('API Version {}'.format(apiVersion.value))

ret = rsa306.DEVICE_Search(byref(numFound), deviceIDs, deviceSerial, deviceType)
#ret = rsa306.Search(deviceIDs, byref(deviceSerial), byref(numFound))

if ret != 0:
    print('Error in Search: ' + str(ret))
    exit()
if numFound.value < 1:
    print('No instruments found. Exiting script.')
    exit()
elif numFound.value == 1:
    print('One device found.')
    print('Device type: {}'.format(deviceType.value))
    print('Device serial number: {}'.format(deviceSerial.value))
    ret = rsa306.DEVICE_Connect(deviceIDs[0])
    if ret != 0:
        print('Error in Connect: ' + str(ret))
        exit()
else:
    print('Unexpected number of devices found, exiting script.')
    exit()
    
cf = c_double(100e6)
ret = rsa306.CONFIG_SetCenterFreq(cf)
if ret != 0:
    print('Error in CONFIG_SetCenterFreq: ' + str(ret))
    exit()
    
    
class IQBLK_ACQINFO(Structure):
        _fields_ = [("sample0Timestamp", c_uint64),
                    ("triggerSampleIndex", c_uint64),
                    ("triggerTimestamp", c_uint64),
                    ("acqStatus", c_uint32)]
    
acqInfo = IQBLK_ACQINFO(1, 1, 1, 1)
class SPECTRUM_SETTINGS(Structure):
        _fields_ = [("span", c_double),
                    ("rbw", c_double),
                    ("enableVBW", c_bool),
                    ("vbw", c_double),
                    ("traceLength", c_int),
                    ("window", c_bool),
                    ("verticalUnit", c_int),
                    ("actualStartFreq", c_double),
                    ("actualStartFreq", c_double),
                    ("actualFreqStepSize", c_double),
                    ("actualRBW", c_double),
                    ("actualVBW", c_double),
                    ("actualNumIQSamples", c_double)]
    
specSet =SPECTRUM_SETTINGS()

class IQSTRMIQINFO(Structure):
        _fields_ = [("timestamp", c_uint64),
                    ("triggerCount", c_int),
                    ("triggerIndices", POINTER(c_int)),
                    ("scaleFactor", c_double),
                    ("acqStatus", c_uint64)]
iqinfo = IQSTRMIQINFO()
    



    
ready = c_bool(False)

def getIQBLKdata():
    aLen = 1280
    aLen = 2560
    aLen = 560000
    iqLen = aLen * 2
    floatArray = c_float * iqLen
    
    length = c_int(aLen)
    ret = rsa306.IQBLK_SetIQRecordLength(length)
    if ret != 0:
        print('Error in SetIQRecordLength: ' + str(ret))
    
    rl = c_double(-10)
    ret = rsa306.CONFIG_SetReferenceLevel(rl)
    if ret != 0:
        print('Error in CONFIG_SetReferenceLevel: ' + str(ret))
 
    iqBW = c_double(40e6)
    ret = rsa306.IQBLK_SetIQBandwidth(iqBW)
    if ret != 0:
        print('Error in IQBLK_SetIQBandwidth: ' + str(ret))
    ret = rsa306.TRIG_SetTriggerMode(0)
    if ret != 0:
        print('Error in TRIG_SetTriggerMode: ' + str(ret))
    
      
    ret = rsa306.DEVICE_Run()
    if ret != 0:
        print("Run error: " + str(ret))
    ret = rsa306.IQBLK_WaitForIQDataReady(10000, byref(ready))
    if ret != 0:
        print("WaitForIQDataReady error: " + str(ret))
    iqData = floatArray()
    startIndex = c_int(0)
    maxBandwidth = c_double(0)
    minBandwidth = c_double(0)
    iqSamplingRate = c_double(0)
    maxCF = c_int(0)
    recordLength = c_int(0)
    
    
    if ready:
        print("Start get Data")
        starttime = time.time()
        previoustimestamp =1
      
        
        for i in range(0,10):
            ret = rsa306.IQBLK_AcquireIQData()
            if ret != 0:
                print("IQBLK_AcquireIQData error: " + str(ret))
            ret = rsa306.TRIG_ForceTrigger()
            print(time.time()-starttime)
            ret = rsa306.IQBLK_WaitForIQDataReady(10000, byref(ready))
            if ret != 0:
                print("WaitForIQDataReady error: " + str(ret))
            print(time.time()-starttime)
            ret = rsa306.IQBLK_GetIQData(byref(iqData), byref(startIndex), length)
            if ret != 0:
                print("GetIQData error: " + str(ret))
            print(i)
            print(time.time()-starttime)
            ret = rsa306.IQBLK_GetIQAcqInfo(byref(acqInfo))
            print("sample0Timestamp: "+str(acqInfo.sample0Timestamp))
            print("triggerSampleIndex: "+str(acqInfo.triggerSampleIndex))
            print("triggerTimestamp: "+str(acqInfo.triggerTimestamp))
            print("acqStatus: "+str(acqInfo.acqStatus))
            ret = rsa306.IQBLK_GetMaxIQBandwidth(byref(maxBandwidth))
            print("Max IQ bandwidth: " + str(maxBandwidth.value))
            ret = rsa306.IQBLK_GetMinIQBandwidth(byref(minBandwidth))
            print("Min IQ bandwidth: " + str(minBandwidth.value))
            ret = rsa306.IQBLK_GetIQSampleRate(byref(iqSamplingRate))
            print("IQ samling rate: " + str(iqSamplingRate.value))
            ret = rsa306.IQBLK_GetMaxIQRecordLength(byref(maxCF))
            print("Max recorder lenght: " + str(maxCF.value))
            ret = rsa306.IQBLK_GetIQRecordLength(byref(recordLength))
            print("IQ data samples at acquisition: " + str(recordLength.value))
            print("Ms to former sampel acquisition: " + str(acqInfo.sample0Timestamp - previoustimestamp))
            print("Delta to triggerpoint" + str(acqInfo.sample0Timestamp-acqInfo.triggerTimestamp ))
            previoustimestamp = acqInfo.sample0Timestamp
        
        print(time.time()-starttime)
        print("Data obtained"+ str(aLen))
        iData = list(range(0,aLen))
        qData = list(range(0,aLen))
        for i in range(0,aLen):
            iData[i] = iqData[i*2]
            qData[i] = iqData[i*2+1]
    
        cf = c_double(0)
        rsa306.CONFIG_GetCenterFreq(byref(cf))
        
        z = [(x + 1j*y) for x, y in zip(iData,qData)]
        spec = np.fft.fft(z, aLen)
        
        spec2 = mlab.specgram(z, NFFT=aLen, Fs=56e6)
        f = [(x + cf)/1e6 for x in spec2[1]]
        r = [x * 1 for x in abs(spec)]
        r = np.fft.fftshift(r)
        return [f,r]

def getSpectrum():
    specSet.span = c_double(40e6)
    specSet.rbw = c_double(30e6)
    specSet.traceLength = c_int(801)
    
    enable = c_bool(True)
    
    rsa306.SPECTRUM_SetEnable(enable)
    rsa306.SPECTRUM_SetDefault()
    rsa306.SPECTRUM_GetSettings(byref(specSet))
    

specData = getIQBLKdata()

plt.plot(specData[0], specData[1])
 # saving:
#f = open("data", "w")
#np.savetxt(f, specData[1])
# loading:
#x, y = np.loadtxt("data", unpack=True)
plt.show()
#r = rsa306.DPX_GetEnable()
rsa306.DEVICE_Stop()

#rsa306.DPX_Configure(enableSpectrum, enableSpectrogram)
rsa306.DEVICE_Disconnect()