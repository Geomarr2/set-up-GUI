# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:15:23 2018
@author: G-man
"""
import matplotlib
matplotlib.use('cairo')
from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
from io import StringIO
from tkinter import *
global fields 
import calData as cal
import os
import pandas as pd
import sys
import time
from scipy.interpolate import spline
#import matplotlib as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.widgets import Lasso
from matplotlib.figure import Figure
FileCCF="CCF4.csv"
PathCCF = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/Spectrum/"
PathGainCircuit = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/RFIChamberMeasure/RFcable/"
FileGainCircuit = "GainCircuit.csv"
fields = 'Path', 'Filename', 'Start Frequency (MHz)', 'Stop Frequency (MHz)', 'LNA gain (dB)', 'Cable losses (dB)', 'Antenna efficiency', 'Set x axes scaling factor', 'Workable bandwidth (MHz)','True bandwidth (MHz)'
AcqBW = 40e6
color = ['y','hotpink','olive','coral','r','b','m','c','g']
bandwidth = AcqBW #Hz
#StartFreq = 1000e6
#StopFreq = 1200e6
Z0 = 119.9169832 * np.pi  # Impedance of freespace
#G_LNA= 20 #dB gain of the LNA
#Lcable = -1  #dB cable losses
#antennaEfficiency = 0.75 
Nsample = 523852
r = 1

# TEST path DataPath = "D:/Geomarr/Spectrum/"       #Path to save the spectra
#TEST filename = 0RFISpectrum
#TEST start_freq = 1000
#TEST stop_freq = 1040

#----------Style options---------#
DEFAULT_PALETTE = {"foreground":"lightblue","background":"black"}
DEFAULT_STYLE_1 = {"foreground":"black","background":"lightblue"}

class NavSelectToolbar(NavigationToolbar2TkAgg): 
    def __init__(self, canvas,root,parent):
        self.canvas = canvas
        self.root   = root
        self.parent = parent
        NavigationToolbar2TkAgg.__init__(self, canvas,root)
        self._zoomrect_default=NavigationToolbar2TkAgg.zoom
        
        self.lasso_button = self._custom_button(text="lasso",command=lambda: self.lasso(
                lambda inds: self.parent.multi_select_callback(inds),"lasso"),**DEFAULT_STYLE_1)
        self.pick_button = self._custom_button(text="select",command=lambda: self.picker(
                lambda ind: self.parent.single_select_callback(ind),"select"),**DEFAULT_STYLE_1)
        
    def _custom_button(self, text, command, **kwargs):
        button = Button(master=self, text=text, padx=2, pady=2, command=command, **kwargs)
        button.pack(side=LEFT,fill="y")
        return button

class GUI_set_up:
    def __init__(self,  window, fields):
        self.Path= "C:/Users/geomarr/Documents/GitHub/set-up-GUI/GUIData/"
        self.Filename = "Spectrum_"
        self.CenterFrequency = 1020*1e6
        self.Bandwidth = 40*1e6
        self.Nchannels = 373851
        self.G_LNA = 39 
        self.Lcable = -1
        self.antennaEfficiency = 0.75
        self.window = window
        self.x_factor = 20
        self.window.title("RFI Chamber")
        
        self.entry_data = self.makeform(fields)
        self.figure = Figure(figsize=(10,10))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        self.canvas.get_tk_widget().place(x = 10, y = 250, width = 250, height = 250)
        self.canvas.get_tk_widget().pack(side=BOTTOM, fill=X, padx=5, pady=5)
        self.fig_plot = self.figure.add_subplot(111)
        self.get_size_plot = self.figure.get_axes()
        self.toolbar = NavSelectToolbar(self.canvas,self.window,self)
        self.toolbar.update()
#        self.CCFdata = self.get_CCF(AcqBW)
        self.GainCircuitData = self.readIQDatafileCSV(PathGainCircuit,FileGainCircuit)
        window.bind('<Return>', (lambda event,e=self.entry_data: self.fetch()))
        button_get = Button(self.window,text = 'accept', command = (lambda e=self.entry_data: self.fetch()))
        button_get.place(x = 10, y = 220)
        button_get.pack()
        button_plot=Button(window,text = 'show plot', command=(lambda e=self.entry_data: self.calibrateData()))
        button_plot.place(x = 60, y = 220)
        button_plot.pack()
        button_plot_clear=Button(window,text = 'clear plot', command=(lambda e=self.entry_data: self.clear_data()))
        button_plot_clear.place(x = 130, y = 220)
        button_plot_clear.pack()
        
    def fetch(self):          
        self.Path= self.entry_data['Path'].get()
        self.Filename = self.entry_data['Filename'].get()
        self.Start_freq = self.entry_data['Start Frequency (MHz)'].get()
        self.Stop_freq = self.entry_data['Stop Frequency (MHz)'].get()
        self.G_LNA = self.entry_data['LNA gain (dB)'].get() 
        self.Lcable = self.entry_data['Cable losses (dB)'].get()
        self.antennaEfficiency = self.entry_data['Antenna efficiency'].get()
        self.get_x_factor = self.entry_data['Set x axes scaling factor'].get()
        temp_workable_BW = self.entry_data['Workable bandwidth (MHz)'].get()
        temp_true_BW = self.entry_data['True bandwidth (MHz)'].get()
        self.workable_BW = float(temp_workable_BW)*1e6
        self.true_BW = float(temp_true_BW)*1e6
        
    def get_data(l):
        l.append(self.entry.get())
        print(l)
        
    def readIQDatafileCSV(self,path, filename):
        data = []
        with open(path+filename, "r") as filedata:
            csvdata = np.genfromtxt(filedata, delimiter = ',')
            for row in csvdata:
                data.append(row)
            filedata.close()
            data =np.array(data, dtype = 'float')    
        return data
        
    def clear_data(self):
        self.canvas.get_tk_widget().delete("all")
        self.fig_plot.clear()
        
    def makeform(self, fields):
       entries = {}
       for field in fields:
           row = Frame(self.window)
           lab = Label(row, width=22, text=field+": ", anchor='w')
           ent = Entry(row, width=22)
           # ent.insert(0,"0")
           row.pack(side=TOP, fill=X, padx=5, pady=5)
           lab.pack(side=LEFT)
           ent.pack(side=RIGHT, expand=NO, fill=X)
           entries[field] = ent 
       return entries
       
    def get_CCF(self): 
       CCFdata = self.readIQDatafileCSV(PathCCF,FileCCF)
       upperfreq = self.CenterFrequency + self.Bandwidth/2 
       lowerfreq =self.CenterFrequency - self.Bandwidth/2
       temp_spec = np.array([], dtype=np.float32)
       temp = 0
       for j in range(len(CCFdata)):
           if lowerfreq <= CCFdata[j,0] and upperfreq > CCFdata[j,0]:
               temp = CCFdata[j,1] 
               temp_spec = np.append(temp_spec, temp)
       return temp_spec.astype('float32')  
   
    def cal_GainCircuit(self, array_size = 40): 
       upperfreq = self.CenterFrequency + self.Bandwidth/2 
       lowerfreq =self.CenterFrequency - self.Bandwidth/2
       temp_gain = np.array([], dtype=np.float32)
       temp = 0
       freqGain = (self.GainCircuitData[:,0])*1e9
       if len(freqGain) > 1:
            count = 0
            for j in range(len(freqGain)):
               if lowerfreq <= freqGain[j] and upperfreq >= freqGain[j]:
                   temp = self.GainCircuitData[j,1] + temp
                   count = count + 1
            temp = temp/count
            temp_gain = np.append(temp_gain, temp)
       else:
           for j in range(len(freqGain)):
               if lowerfreq <= freqGain[j] and upperfreq >= freqGain[j]:
                   temp = self.GainCircuitData[j,1] 
                   temp_gain = np.append(temp_gain, temp)
       return (temp_gain.T).astype('float32') 
   
    def read_reduced_Data(self, path, filename, start_freq, stop_freq, array_size = 40, resfact=1):
        # 40 MHz kom hier in
        temp_spec = np.array([])
        temp_freq = np.array([])
        spectrum =np.load(path + filename)
        x = int(len(spectrum)/array_size)
        for i in range(int(array_size)):
        #trimspec = np.array(self.change_freq_channel(self.trim_spectrum(spectrum),resfact))
            temp = np.max(spectrum[i*x:x*(i+1),1])
            tempfreq = np.mean(spectrum[i*x:x*(i+1),0])
            temp_spec = np.append(temp_spec, temp)
            temp_freq = np.append(temp_freq, tempfreq)
        temp = temp_freq, temp_spec
        y = np.array(temp, dtype=np.float32)
        return y
    
    def reduced_Data(self,spectrum, array_size = 40, resfact=1):
        # 40 MHz kom hier in
        # array size is the number of point one batch of data point should be
        spectrum 
        temp_spec = np.array([])
        temp_freq = np.array([])
        # number of samples
        nr_smp = int(((self.Stop_freq*1e6-self.Start_freq*1e6)/self.workable_BW)*array_size)
        
        x = int(len(spectrum)/nr_smp)
        for i in range(nr_smp):
        #trimspec = np.array(self.change_freq_channel(self.trim_spectrum(spectrum),resfact))
            temp = np.mean(spectrum[i*x:x*(i+1),1])
            tempfreq = np.mean(spectrum[i*x:x*(i+1),0])
            temp_spec = np.append(temp_spec, temp)
            temp_freq = np.append(temp_freq, tempfreq)
        temp = temp_freq, temp_spec
        y = np.array(temp, dtype=np.float32)
        return y
    def get_x(self):
        # GET X AXIS VALUES (MAX AND MIN) FROM ZOOM????????
        #display_BW = X_max - X_min self.Stop_freq*1e6-self.Start_freq*1e6
        if display_BW <= 5000*1e6 and display_BW > 3000*1e6:
            array_size = int(self.workable_BW/100e3)
        if display_BW <= 3000*1e6 and display_BW > 500*1e6:
            array_size = int(self.workable_BW/50e3)
        else:
            # set full resolution
            array_size = int(self.workable_BW/self.resolution)

    def read_reduce_Data(self):
        # read in the whole BW in one array
        spec = np.array([])
        freq = np.array([])
        path = self.Path
        Stop_freq = self.CenterFrequency + self.Bandwidth/2
        Start_freq = self.CenterFrequency - self.Bandwidth/2
        #res = self.true_BW/self.Nsample
# set the display sample size depending on the display bandwidth and resolution 
        new_resolution = 100e3
        array_size = 40
        Freqlist =cal.generate_CFlist(int(Start_freq),int(Stop_freq))
        print(Freqlist)
        for i in Freqlist: # i = 40 Mhz range
            filename = self.Filename+str(i/1e6)+"MHz.npy"
            spectrum_temp = np.load(path + filename)
            freq =  spectrum_temp[:,0].T*1e6
            spectrum =  spectrum_temp[:,1].T
            print(len(freq))
            x = int(len(freq)/array_size)
            for i in range(array_size):
                temp_spec = np.max(spectrum[i*x:x*(i+1)])
                spec = np.append(spec, temp_spec)
        freq = np.linspace(Start_freq,Stop_freq,array_size)
        temp = freq,spec
        y = np.array(temp, dtype=np.float32)
        return y
    
    def readSpectralData(self,path,filename):
        arraydata = np.load(path + filename)
        return arraydata
    
    def calCCF(self,spectrum, CCF, r, Lcable, G_LNA, antennaEfficiency): # returns in [dBuV/m]
         # spectrum numpy array
         #get the 1MHz average of the spectrum
         temp = -G_LNA - Lcable - (10.0 * np.log10(antennaEfficiency)) - CCF + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r*r)))) + 90.0
         
         return spectrum[0], spectrum[1]+temp
               
       
    def calibrateData(self):
       tstart = time.time()
       temp_maxSpec = []
       temp_minSpec = []
       if len(self.entry_data) == 0:
           print('Invalid directory defined')
       else:
           G_LNA = int(self.G_LNA) 
           G_LNA = self.cal_GainCircuit()
           Lcable = int(self.Lcable)
           antennaEfficiency = np.float32(self.antennaEfficiency) 
           spectrum = self.read_reduce_Data()
           #spec = self.reduced_Data(spectrum_whole_bw,40)
           #smooth_spectrum_whole_bw = smooth(spectrum_whole_bw[1])
           CCF = self.get_CCF()
           #Spec = self.calCCF(spec, CCF, r, int(Lcable), G_LNA, float(antennaEfficiency))
           #temp_maxSpec.append(max(Spec[1]))
           #temp_minSpec.append(min(Spec[1]))
           cal_spec = self.fig_plot.plot(spectrum[0]/1e6,spectrum[1], color=color[0])#,label = 'Calibarted data')
           #raw_spec, = self.fig_plot.plot(spec[0]/1e6,spec[1], color=color[1],label='Raw data')
           #self.fig_plot.legend()#([cal_spec,raw_spec],['Calibarted data', 'Raw data'])
#           maxSpec = max(temp_maxSpec)
#           minSpec = min(temp_minSpec)
#           self.fig_plot.axis([int(self.Start_freq),int(self.Stop_freq), minSpec-self.x_factor, maxSpec+self.x_factor])
           self.fig_plot.set_ylabel("Electrical Field Strength [dBuV/m]")#('Power [dBm]')
           self.fig_plot.set_xlabel("Frequency (MHz) (resolution %.3f kHz)"%1)
#           print('Path: %s \nFilename: %s \nStart Frequency (MHz): %s \nStop Frequency (MHz): %s \nLNA gain (dB): %s\nCable losses (dB): %s \nAntenna Efficiency: %s \n' % (self.Path, self.Filename, self.Start_freq, self.Stop_freq, self.G_LNA, self.Lcable, self.antennaEfficiency)) 
           self.canvas.draw()
                   #tempfreq = Spec[:,0]/1e6
                   #temp = Spec[:,1]
                   #temp_spec = np.append(temp_spec, temp)
                   #temp_freq = np.append(temp_freq, tempfreq)able, G_LNA, antennaEfficiency)
                   #self.fig_plot.plot(x[:,0],x[:,1], c=color[1])
                   #self.fig_plot.plot(Spec[0],Spec[1], c=color[1])
    #           maxSpec = max(temp_maxSpec)
    #           minSpec = min(temp_minSpec)
    #           self.fig_plot.axis([int(self.Start_freq),int(self.Stop_freq), minSpec+self.x_factor, maxSpec-self.x_factor])

       print ('time:' , (time.time()-tstart))
if __name__ == '__main__':
   window = Tk()
   start = GUI_set_up(window, fields)
   window.mainloop() 