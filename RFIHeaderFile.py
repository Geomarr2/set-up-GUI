# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 10:59:30 2018

@author: geomarr
"""

import ctypes as c
import os
import numpy as np
import math as math
import matplotlib.pyplot as plt
import DataLoad

class HeaderInformation(object):
    def __init__(self):
        self.centerFrequency = 0.
        self.bandwidth = 0.
        self.nSample = 0
        self.frequencySpacing = 'uniform'
        self.integrationTime =  0.
        self.scanID  = 0
        self.acquisitionSystem = 'default'
        self.chambercalib = 'default'
        self.antennacalib  = 'default'
        self.cablecalib = 'default'
        self.lnaCalib = 'default'
        self.backgroundData = 'default'
        self.timestamp = 'default'
        self.dutInformation  = 'default'
        self.fileprefix ='RFIscan'
        self.filename = 'default'
        self.path = os.getcwd()
        self.Stop_freq = 0.
        self.Start_freq = 0.
        DataLoad.DataInfo.__init__(self)
        
        
    def createFilename(self, directory  = os.getcwd()):
        if self.centerFrequency == 0:
            _log.warning("Center frequency missing while creating header file!")
        if self.timestamp == 'default':
            _log.info("Setting timestamp for header file")
            self.setTimestamp()
        strtime =  self.timestamp.replace(":","")
        strtime =  strtime.replace("-","")
        strtime =  strtime.replace(".","")
        
        self.filename =  self.fileprefix +"_" + str(int(self.centerFrequency)) + "_" +strtime + ".rfi"
    
         
    def saveToFile(self):
        if self.filename =='default':
            _log.info("Creating default header filename")
            self.createFilename()
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        newFile = open(str(self.path +"/"+self.filename),"w")
        newFile.write(self.createHeaderString())
        newFile.close()
              
    def findCenterFrequency(self,HeaderFiles):
#Load in all the header file get the info and the BW from max and min center freq
        #HeaderFiles = DataLoad.DataInfo
        center_freq_temp = []
        for i in range(len(HeaderFiles)): # i = 40 Mhz range
            temp_path = HeaderFiles[i].split('_')
            name_file = temp_path[0]
            cfreq = temp_path[1]
            print(temp_path[1])
            scanID_file = temp_path[2]
            center_freq_temp.append(float(cfreq))
            
        self.Stop_freq = max(center_freq_temp) + self.Bandwidth/2
        self.Start_freq = min(center_freq_temp) - self.Bandwidth/2    
        print('Start freqeuncy (Hz): %f \nStop freqeuncy (Hz): %f'%(self.Start_freq, self.Stop_freq))
        
    def printHeaderInfo(self):
#Load in all the header file get the info and the BW from max and min center freq
        HeaderFiles = DataLoad.DataInfo.headerFile
        with open(HeaderFiles, "r") as file:
            for line in file:
                print (line.split("'")[1])
        
    def genScanID(self):
        IDStr = str(self.acquisitionSystem) + str(time.time())
        IDStr = IDStr.replace(".","")
        IDStr = IDStr.replace(":","")
        IDStr = IDStr.replace("_","")
        self.scanID = IDStr
        
    def createHeaderString(self):
        strdata = "MPIfR RFI file \n\nDATA"
        data = self.exportHeaderInfoinList()
        for x, y in zip(data[0], data[1]):
            strdata +="\n"+ str(y)+":\n"+str(x)
        return strdata
    
    def getHeaderInfoinList(self):
        data = [self.centerFrequency,
                         self.bandwidth,
                         self.nSample,
                         self.frequencySpacing,
                         self.integrationTime,
                         self.scanID,
                         self.acquisitionSystem,
                         self.chambercalib,
                         self.antennacalib,
                         self.cablecalib,
                         self.lnaCalib,
                         self.backgroundData,
                         self.timestamp,
                         self.dutInformation]
        description = ["Center Frequency in Hz",
                       "Bandwidth in Hz",
                       "Number of Samples",
                       "Frequency Spacing",
                       "Integration time in milliseconds", 
                       "Unique Scan ID",
                       "Data Acquisition System", 
                       "Chamber Calibration",
                       "Antenna Calibration", 
                       "Cable Calibration",
                       "LNA Calibration",
                       "Background Data",
                       "Timestamp",
                       "DUT comments"]
        return list([data, description])       