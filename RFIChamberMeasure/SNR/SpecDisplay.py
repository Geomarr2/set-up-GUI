# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:22:46 2018

@author: User
"""

"""
Peak Power Detect using API for RSA306
Author: Morgan Allison
Date: 6/24/15
Windows 7 64-bit
Python 2.7.9 64-bit (Anaconda 3.7.0)
NumPy 1.8.1, MatPlotLib 1.3.1
"""

from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
"""
################################################################
C:\Tektronix\RSA306 API\lib\x64 needs to be added to the 
PATH system environment variable
################################################################
"""

#\\ required in path because \xhh is the Python representation of a hex value
# Import the RSA300API DLL
os.chdir("C:\\Tektronix\\RSA_API\\lib\\x64")
rsa300 = ctypes.WinDLL("RSA_API.dll")
#create Spectrum_Settings data structure
class Spectrum_Settings(Structure):
   _fields_ = [('span', c_double), ('rbw', c_double),
   ('enableVBW', c_bool), ('vbw', c_double),
   ('traceLength', c_int), ('window', c_int),
   ('verticalUnit', c_int), ('actualStartFreq', c_double), ('actualStopFreq', c_double),
   ('actualFreqStepSize', c_double), ('actualRBW', c_double),
   ('actualVBW', c_double), ('actualNumIQSamples', c_double)]

#initialize variables
specSet = Spectrum_Settings()
longArray = c_long*10
deviceIDs = longArray()
deviceSerial = c_wchar_p('')  
numFound = c_int(0)
enable = c_bool(True)         #spectrum enable
cf = c_double(2.4453e9)       #center freq
refLevel = c_double(0)        #ref level
ready = c_bool(False)         #ready
timeoutMsec = c_int(500)      #timeout
trace = c_int(0)              #select Trace 1 
detector = c_int(1)           #set detector type to max

#search the USB 3.0 bus for an RSA306
ret = rsa300.Search(deviceIDs, byref(deviceSerial), byref(numFound))
if ret != 0:
   print('Error in Search: ' + str(ret))
if numFound.value < 1:
   print('No instruments found.')
   sys.exit()
elif numFound.value == 1:
   print('One device found.')
   print('Device Serial Number: ' + deviceSerial.value)
else:
   print('2 or more instruments found.')
   #note: the API can only currently access one at a time

#connect to the first RSA306
ret = rsa300.Connect(deviceIDs[0])
if ret != 0:
   print('Error in Connect: ' + str(ret))

#preset the RSA306 and configure spectrum settings
rsa300.Preset()
rsa300.SetCenterFreq(cf)
rsa300.SetReferenceLevel(refLevel)
rsa300.SPECTRUM_SetEnable(enable)
rsa300.SPECTRUM_SetDefault()
rsa300.SPECTRUM_GetSettings(byref(specSet))

#configure desired spectrum settings
#some fields are left blank because the default
#values set by SPECTRUM_SetDefault() are acceptable
specSet.span = c_double(40e6)
specSet.rbw = c_double(30e3)
#specSet.enableVBW = 
#specSet.vbw = 
specSet.traceLength = c_int(801)
#specSet.window = 
#specSet.verticalUnit = 
#specSet.actualStartFreq = 
#specSet.actualFreqStepSize = 
#specSet.actualRBW = 
#specSet.actualVBW = 
#specSet.actualNumIQSamples = 

#set desired spectrum settings
rsa300.SPECTRUM_SetSettings(specSet)

#uncomment this if you want to print out the spectrum settings
"""
#print out spectrum settings for a sanity check
print('Span: ' + str(specSet.span))
print('RBW: ' + str(specSet.rbw))
print('VBW Enabled: ' + str(specSet.enableVBW))
print('VBW: ' + str(specSet.vbw))
print('Trace Length: ' + str(specSet.traceLength))
print('Window: ' + str(specSet.window))
print('Vertical Unit: ' + str(specSet.verticalUnit))
print('Actual Start Freq: ' + str(specSet.actualStartFreq))
print('Actual End Freq: ' + str(specSet.actualStopFreq))
print('Actual Freq Step Size: ' + str(specSet.actualFreqStepSize))
print('Actual RBW: ' + str(specSet.actualRBW))
print('Actual VBW: ' + str(specSet.actualVBW))
"""
#initialize variables for GetTrace
traceArray = c_float * specSet.traceLength
traceData = traceArray()
outTracePoints = c_int()

#generate frequency array for plotting the spectrum
freq = np.arange(specSet.actualStartFreq, 
   specSet.actualStartFreq + specSet.actualFreqStepSize*specSet.traceLength, 
   specSet.actualFreqStepSize)

#start acquisition
rsa300.Run()
while ready.value == False:
   rsa300.SPECTRUM_WaitForDataReady(timeoutMsec, byref(ready))
print('Trace Data is Ready')

rsa300.SPECTRUM_GetTrace(c_int(0), specSet.traceLength, 
   byref(traceData), byref(outTracePoints))
print('Got trace data.')
rsa300.Stop()

#convert trace data from a ctypes array to a numpy array
trace = np.ctypeslib.as_array(traceData)

#Peak power and frequency calculations
peakPower = np.amax(trace)
peakPowerFreq = freq[np.argmax(trace)]
print('Peak power in spectrum: %4.3f dBm @ %d Hz' % (peakPower, peakPowerFreq))

#plot the spectrum trace (optional)
plt.plot(freq, traceData)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dBm)')

#BONUS clean up plot axes
xmin = np.amin(freq)
xmax = np.amax(freq)
plt.xlim(xmin,xmax)
ymin = np.amin(trace)-10
ymax = np.amax(trace)+10

plt.show()