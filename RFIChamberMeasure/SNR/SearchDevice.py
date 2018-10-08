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
    
    
#data transfer variables
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

rsa306.DEVICE_Run()
ready = c_bool(False)

#check data ready 
while ready.value == False:
    rsa306.IQBLK_WaitForIQDataReady(timeoutMsec, byref(ready))
    
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

rsa306.DEVICE_Stop()

#rsa306.DPX_Configure(enableSpectrum, enableSpectrogram)
rsa306.DEVICE_Disconnect()
    