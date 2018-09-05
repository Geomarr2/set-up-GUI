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
import pandas as pd

FileCCF="CCF4.csv"
PathCCF = "D:/Geomarr/Spectrum/"
fields = 'Path', 'Filename', 'Start Frequency (MHz)', 'Stop Frequency (MHz)', 'LNA gain (dB)', 'Cable losses (dB)', 'Antenna efficiency'
AcqBW = 40e6
color = ['y','hotpink','olive','coral','darkkhaki','orchid','lightblue','navy','rosybrown','cornflowerblue','lavenderblush','cadetblue','hotpink','olive','coral','darkkhaki','orchid','lightblue','navy','aliceblue','r','b','m','c','g','y','hotpink','olive','coral','darkkhaki','orchid','lightblue','navy','rosybrown','cornflowerblue','lavenderblush','cadetblue','hotpink','olive','coral','darkkhaki','orchid','lightblue','navy','aliceblue','r','b','m','c','g','y','hotpink','olive','coral','darkkhaki','orchid','lightblue','navy','rosybrown','cornflowerblue','lavenderblush','cadetblue','hotpink','olive','coral','darkkhaki','orchid','lightblue','navy','aliceblue','r','b','m','c','g']
bandwidth = AcqBW #Hz
#StartFreq = 1000e6
#StopFreq = 1200e6
Z0 = 119.9169832 * np.pi  # Impedance of freespace
#G_LNA= 20 #dB gain of the LNA
#Lcable = -1  #dB cable losses
#antennaEfficiency = 0.75 
sample = 523852
r = 1
CCFdata = cal.readIQDatafileCCF(PathCCF,FileCCF)
# TEST path DataPath = "D:/Geomarr/Spectrum/"       #Path to save the spectra
#TEST filename = 0RFISpectrum
#TEST start_freq = 1000
#TEST stop_freq = 1040


    
def fetch(data_enter):
    Path= data_enter['Path'].get()
    Filename = ents['Filename'].get()
    Start_freq = data_enter['Start Frequency (MHz)'].get()
    Stop_freq = data_enter['Stop Frequency (MHz)'].get()
    G_LNA = data_enter['LNA gain (dB)'].get() 
    Lcable = data_enter['Cable losses (dB)'].get()
    antennaEfficiency = data_enter['Antenna efficiency'].get()
    print('Path: %s \nFilename: %s \nStart Frequency (MHz): %s \nStop Frequency (MHz): %s \nLNA gain (dB): %s\nCable losses (dB): %s \nAntenna Efficiency: %s \n' % (Path, Filename, Start_freq, Stop_freq, G_LNA, Lcable, antennaEfficiency)) 
   
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
   start_freq = int(data_enter['Start Frequency (MHz)'].get())*1e6
   stop_freq = int(data_enter['Stop Frequency (MHz)'].get())*1e6
   Freqlist =cal.generate_CFlist(start_freq,stop_freq)
   Length_freq = len(Freqlist)
   F = []
   tempCCFData = []
   tempSpec = []
   # Average CCF for each given centre frequency and AcqBW
   #upperfreq = start_freq + AcqBW/2 
   #lowerfreq = stop_freq - AcqBW/2
# get the spectrum
   G_LNA = int(data_enter['LNA gain (dB)'].get()) 
   Lcable = int(data_enter['Cable losses (dB)'].get())
   antennaEfficiency = float(data_enter['Antenna efficiency'].get())
   path = data_enter['Path'].get() +"_gLNA"+str(G_LNA)+"_Lcable"+str(Lcable)+"_EffAnt"+str(antennaEfficiency)+"/"
   path = data_enter['Path'].get()
   filename = data_enter['Filename'].get()  

   # Extract the CCF for the correct freqeuncy range and get the raw spectrum data in one array 
   for i in Freqlist:
       temp = 0
       count = 0
       fileName = filename+str(int(i/1e6))+"MHz.npy"
       F.append(fileName)
       for j in range(len(CCFdata)):
           upperfreq = i + AcqBW/2 
           lowerfreq = i - AcqBW/2
           if lowerfreq <= CCFdata[j][0] and upperfreq >= CCFdata[j][0]:
               temp = CCFdata[j][1] + temp# Chamber Calibration Factor in dBm
               count +=1
       CCFavg = temp/count
       tempCCFData.append(CCFavg)
   F = np.array(F, dtype = 'str')         
   tempCCFData = np.array(tempCCFData, dtype = 'float')  
   
   # Calibrate the raw spectrum
   for i in range(Length_freq):
       fileName = F[i]
       centerFreq = Freqlist[i] #Hz
       CCF = tempCCFData[i]
       ReadFile = cal.readIQDataBin(path,fileName)
       Spec = cal.calCCF(ReadFile, CCF, r, Lcable, G_LNA, antennaEfficiency)
       tempSpec.append(Spec)
       
   # Plot the calibrated data         
   tempSpec = np.array(tempSpec, dtype = 'float')
   print(Length_freq)
   for i in range(Length_freq):
       cal.plot_stiched_spectrum(tempSpec[i,:,:],color[i])         

if __name__ == '__main__':
   my_window = Tk()
   my_window.wm_title("RFI Chamber")
   ents = makeform(my_window, fields)
   my_window.bind('<Return>', (lambda event, e=ents: fetch(e)))
                         # The lambda function used here takes one argument, 
                                                                                # and returns None
   b1=Button(my_window,text = 'accept', command=(lambda e=ents: fetch(e)))
   b1.pack(side=LEFT, padx=5, pady=5)
   #root.bind('<Return>',  cal.plot_stiched_spectrum(Spec,color[1]))
   #cal_data(data_enter, CCFdata)
   #root.bind('<Return>', (lambda event, e=ents: fetch(e)))
   #Spec = cal_data(ents, CCFdata)
   
   b2=Button(my_window,text = 'show plot', command=(lambda e=ents: cal_data(e)))
   b2.pack(side=LEFT, padx=5, pady=5)
  # root.bind('<Return>',  cal.plot_stiched_spectrum(Spec,color[1]))
   my_window.mainloop() 
   