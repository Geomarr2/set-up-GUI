# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:59:40 2018

@author: User
"""


import ctypes as c
import os
import numpy as np
import math as math
import matplotlib.pyplot as plt
import time,datetime
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
_log = logging.getLogger("spectrometer")
_log.setLevel("INFO")


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
        self.genScanID()
        self.dataFile = []
        self.headerFile = []
        self.CCF = []
        self.G_lna = []
        
        
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
    
    
    def setTimestamp(self, localtimestamp = datetime.datetime.now()):
        self.timestamp = localtimestamp.isoformat()
         
    def saveToFile(self):
        if self.filename =='default':
            _log.info("Creating default header filename")
            self.createFilename()
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        newFile = open(str(self.path +"/"+self.filename),"w")
        newFile.write(self.createHeaderString())
        newFile.close()

#send file for file
    def readFromFile(self,filename):
        f = open(filename,"r")
        lines = f.readlines()
        f.close()
        data = self.exportHeaderInfoinList()
#        [for cnt, v in enumerate(data[1]) 
        for counter, value in enumerate(lines): 
            for cnt, v in enumerate(data[1]):
                vf = v+':\n'
                if vf==value:
                    data[0][cnt] = lines[counter+1]
        return data

        
    def loadDataFromFile(self,Start_freq,Stop_freq,dataFile):
        spec = np.array([], dtype=np.float32)
       # spec = []
       # spec = [np.load(dataFile[i]) for i in range(len(dataFile))]
       # print(spec)
        for i in range(len(dataFile)):
            temp = np.load(dataFile[i])
            spec = np.append(spec, temp)
        freq = np.linspace(Start_freq,Stop_freq,len(spec))
        temp = freq,spec
       # self.DATA = np.array(spec, dtype=np.float32) 
        return np.array(temp, dtype=np.float32) 
    
    def genScanID(self):
        IDStr = str(self.acquisitionSystem) + str(time.time())
        IDStr = IDStr.replace(".","")
        IDStr = IDStr.replace(":","")
        IDStr = IDStr.replace("_","")
        self.scanID = IDStr
        
    def getDataID(self,headerFile):
#Load in all the header file get the info and the BW from max and min center freq
        
        headerInfo = [open(headerFile[count],'r').readlines() for count, val in enumerate(headerFile)]
   
        tempID = [value[Cnt+1] for counter, value in enumerate(headerInfo) for Cnt, Val in enumerate(value) if Val == 'Unique Scan ID:\n']
        tempID = [value[:-1] for counter, value in enumerate(tempID) if value.endswith('\n')]
        return tempID
        
    def createHeaderString(self):
        strdata = "MPIfR RFI file \n\nDATA"
        data = self.exportHeaderInfoinList()
        for x, y in zip(data[0], data[1]):
            strdata +="\n"+ str(y)+":\n"+str(x)
        return strdata
    
    def exportHeaderInfoinList(self):
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
        

    