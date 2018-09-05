# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:15:23 2018

@author: G-man

"""
import numpy as np
from tkinter import *
global fields 
import calData as cal
import os

FileCCF="CCF4.csv"
PathCCF = "D:/Geomarr/Spectrum/"
fields = 'Path', 'Filename', 'Start Frequency (MHz)', 'Stop Frequency (MHz)'
AcqBW = 40e6
bandwidth = AcqBW #Hz
StartFreq = 1000e6
StopFreq = 1200e6
Z0 = 119.9169832 * np.pi  # Impedance of freespace
G_LNA= 10 #dB gain of the LNA
Lcable = -1  #dB cable losses
antennaEfficiency = 0.75 
sample = 523852
r = 1
CCFdata = cal.readIQDatafileCCF(PathCCF,FileCCF)
# TEST path DataPath = "D:/Geomarr/Data/"       #Path to save the spectra
#TEST filename = Spec
#TEST start_freq = 1000
#TEST stop_freq = 1040


    
def fetch(ents):
    Path= ents['Path'].get()
    Filename = ents['Filename'].get()
    Start_freq = ents['Start Frequency (MHz)'].get()
    Stop_freq = ents['Stop Frequency (MHz)'].get()
    print('Path: %s \nFilename: %s \nStart Frequency (MHz): %s \nStop Frequency (MHz): %s \n' % (Path, Filename, Start_freq, Stop_freq)) 
   
def get_data(l):
    l.append(box1.get())
    print(l)
    
def makeform(root, fields):
   entries = {}
   for field in fields: 
      row = Frame(root)
      lab = Label(row, width=22, text=field+": ", anchor='w')
      ent = Entry(row)
     # ent.insert(0,"0")
      row.pack(side=TOP, fill=X, padx=5, pady=5)
      lab.pack(side=LEFT)
      ent.pack(side=RIGHT, expand=YES, fill=X)
      entries[field] = ent 
   return entries
   
def cal_data(data_enter):
   start_freq = float(data_enter['Start Frequency (MHz)'].get())*1e6
   stop_freq = int(data_enter['Stop Frequency (MHz)'].get())*1e6
   # Average CCF for each given centre frequency and AcqBW
   upperfreq = start_freq + AcqBW/2 
   lowerfreq = stop_freq - AcqBW/2
# get the spectrum
   path = data_enter['Path'].get() 
   filename = data_enter['Filename'].get() + str(int(upperfreq/1e6))+"MHz.npy" 
   
   ReadFile = cal.readIQDataBin(path,filename)
   for j in range(len(CCFdata)):
       if lowerfreq <= CCFdata[j][0] and upperfreq >= CCFdata[j][0]:
           CCFavg = CCFdata[j][1] # Chamber Calibration Factor in dBm
           
   Spec = cal.calCCF(ReadFile, CCFavg, r, Lcable, G_LNA, antennaEfficiency) 
   cal.plot_stiched_spectrum(Spec[1],cal.color[1])
   
  
   #return Spec 
if __name__ == '__main__':
   root = Tk()
   root.wm_title("RFI Chamber")
   ents = makeform(root, fields)
   root.bind('<Return>', (lambda event, e=ents: fetch(e)))
                         # The lambda function used here takes one argument, 
                                                                                # and returns None
   b1=Button(root,text = 'accept', command=(lambda e=ents: fetch(e)))
   b1.pack(side=LEFT, padx=5, pady=5)
   #root.bind('<Return>',  cal.plot_stiched_spectrum(Spec,color[1]))
   #cal_data(data_enter, CCFdata)
   #root.bind('<Return>', (lambda event, e=ents: fetch(e)))
   #Spec = cal_data(ents, CCFdata)
   
   b2=Button(root,text = 'show plot', command=(lambda e=ents: cal_data(e)))
   b2.pack(side=LEFT, padx=5, pady=5)
   #root.bind('<Return>',  cal.plot_stiched_spectrum(Spec,color[1]))
   root.mainloop() 
   