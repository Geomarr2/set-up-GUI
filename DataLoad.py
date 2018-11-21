# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:14:45 2018

@author: geomarr
"""

import ctypes as c
import os
import numpy as np
import math as math
import matplotlib.pyplot as plt
import RFIHeaderFile

class DataInfo(object):
    def __init__(self):
        self.headerFile = np.array([])
        self.dataFile = np.array([])

    def dump_filenames(self):
#Load in vereything and seperate header and data file
        global headerFile
        path = self.directory_entry.get()
        path.replace("\\","/") + "/"
        new_path = path.split(' ')
        for i in range(len(new_path)): 
            if new_path[i].endswith(".rfi"):
                self.headerFile = np.append(self.headerFile, new_path[i])
            elif new_path[i].endswith(".npy"):
                self.dataFile = np.append(self.dataFile, new_path[i])
        print('Header filename: %s\nData filename: %s'% (self.headerFile, self.dataFile))
        headerFile = self.headerFile