# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:15:23 2018
@author: G-man
"""
import matplotlib
#matplotlib.use('cairo')
matplotlib.use('TkAgg')
from IPython import get_ipython
get_ipython().magic('reset -sf')
#from bokeh.plotting import figure, output_file, show
import numpy as np
from tkinter import *
global fields 
import calData as cal
import os
import pandas as pd
import sys
import time
import csv
from memory_profiler import profile

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.widgets import Lasso
from matplotlib.figure import Figure



FileCCF="CCF4.csv"
PathCCF = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/Spectrum/"
PathGainCircuit = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/RFIChamberMeasure/RFcable/"
FileGainCircuit = "GainCircuit.csv"
fields = 'Path', 'Filename', 'Start Frequency (MHz)', 'Stop Frequency (MHz)', 'Cable losses (dB)', 'Antenna efficiency'
default = ["C:/Users/geomarr/Documents/GitHub/set-up-GUI/Spectrum/", "0RFISpectrum", 1000, 1040, -1, 0.75]
AcqBW = 40e6
color = ['y','hotpink','r','b','m','c','g']
bandwidth = AcqBW #Hz
#StartFreq = 1000e6
#StopFreq = 1200e6
Z0 = 119.9169832 * np.pi  # Impedance of freespace
#G_LNA= 20 #dB gain of the LNA
#Lcable = -1  #dB cable losses
#antennaEfficiency = 0.75 
sample = 523852
r = 1

# TEST path DataPath = "D:/Geomarr/Spectrum/"       #Path to save the spectra
#TEST filename = 0RFISpectrum
#TEST start_freq = 1000
#TEST stop_freq = 1040

#----------Style options---------#
DEFAULT_PALETTE = {"foreground":"lightblue","background":"black"}
DEFAULT_STYLE_1 = {"foreground":"black","background":"lightblue"}
precision = 100
fp = open('memory_consumption_GUI.log', 'w')
 
class NavSelectToolbar(NavigationToolbar2TkAgg): 
    def __init__(self, canvas,root,parent):
        self.canvas = canvas
        self.root   = root
        self.parent = parent
        NavigationToolbar2TkAgg.__init__(self, canvas,root)
        self.lasso_button = self._custom_button(text="lasso",command=lambda: self.lasso(
                lambda inds: self.parent.multi_select_callback(inds),"lasso"),**DEFAULT_STYLE_1)
        self.pick_button = self._custom_button(text="select",command=lambda: self.picker(
                lambda ind: self.parent.single_select_callback(ind),"select"),**DEFAULT_STYLE_1)
        
    def _custom_button(self, text, command, **kwargs):
        button = Button(master=self, text=text, padx=2, pady=2, command=command, **kwargs)
        button.pack(side=LEFT,fill="y")
        return button
@profile(precision=precision, stream=fp)
class GUI_set_up:
    def __init__(self,  window, fields):
        self.window = window
        window.title("RFI Chamber")
        entry = self.makeform(fields)
        figure = Figure(figsize=(10,10))
        self.canvas = FigureCanvasTkAgg(figure, master=window)
        self.canvas.get_tk_widget().place(x = 10, y = 250, width = 250, height = 250)
        self.canvas.get_tk_widget().pack(side=BOTTOM, fill=X, padx=5, pady=5)
        self.fig_plot = figure.add_subplot(111)
        toolbar = NavSelectToolbar(self.canvas,window,self)
        toolbar.update()
        self.CCFdata = self.readIQDatafileCSV(PathCCF,FileCCF)
        self.GainCircuitData = self.readIQDatafileCSV(PathGainCircuit,FileGainCircuit)
        #window.bind('<Return>', (lambda event,e=self.entry_data: self.fetch()))
        #button_get = Button(window,text = 'accept', command = (lambda e=self.entry_data: self.fetch()))
        #button_get.place(x = 10, y = 220)
        #button_get.pack()
        window.bind('<Return>', (lambda event,e=entry: self.fetch(e)))
        button_get = Button(window,text = 'accept', command = (lambda e=entry: self.fetch(e)))
        button_get.place(x = 10, y = 220)
        button_get.pack()
        button_get = Button(window,text = 'set default', command = (lambda e=entry: self.check_entry_data_empty(e)))
        button_get.place(x = 10, y = 220)
        button_get.pack()
        button_plot= Button(window,text = 'show plot', command=(lambda e=entry: self.cal_data(e)))
        button_plot.place(x = 60, y = 220)
        button_plot.pack()
        #button_plot_clear= Button(window,text = 'clear plot', command=(lambda e=self.entry_data: self.clear_data()))
        #button_plot_clear.place(x = 130, y = 220)
        #button_plot_clear.pack()
        
    def check_entry_data_empty(self,entry_data):          
        entry_data['Path'].insert(0,"C:/Users/geomarr/Documents/GitHub/set-up-GUI/Spectrum/")
        entry_data['Filename'].insert(0,"0RFISpectrum")
        entry_data['Start Frequency (MHz)'].insert(0,"1000")
        entry_data['Stop Frequency (MHz)'].insert(0,"1040")
        entry_data['Cable losses (dB)'].insert(0,"-1")
        entry_data['Antenna efficiency'].insert(0,"0.75")
        return entry_data
        
    def readIQDatafileCSV(self,path, filename):
        data = []
        with open(path+filename, "r") as filedata:
            csvdata = csv.reader(filedata, delimiter = ',')
            for row in csvdata:
                data.append(row)
            filedata.close()
            data =np.array(data, dtype = 'float')    
        return data

    def fetch(self, entry):          
        Path= entry['Path'].get()
        Filename = entry['Filename'].get()
        Start_freq = entry['Start Frequency (MHz)'].get()
        Stop_freq = entry['Stop Frequency (MHz)'].get()
        Lcable = entry['Cable losses (dB)'].get()
        antennaEfficiency = entry['Antenna efficiency'].get()
        return Path, Filename, Start_freq, Stop_freq, Lcable, antennaEfficiency
    
    def delete_data(self,entry):
        entry.delete(0, END)

        
    def clear_data(self):
        self.canvas.get_tk_widget().delete("all")
        self.fig_plot.clear()
        
    def makeform(self, fields):
       entries = {}
       for num, field in enumerate(fields):
           #var.set(default[num])
           row = Frame(self.window)
           lab = Label(row, width=22, text=field+": ", anchor='w')
           ent = Entry(row, width=22)
           #ent.insert(num, str(default[num]))
           row.pack(side=TOP, fill=X, padx=5, pady=5)
           lab.pack(side=LEFT)
           ent.pack(side=RIGHT, expand=NO, fill=X)
           entries[field] = ent 
       return entries
          
    def cal_CCF(self, freq): 
       upperfreq = freq + AcqBW/2 
       lowerfreq = freq - AcqBW/2
       points = int((upperfreq - lowerfreq)/1e6)
       temp_spec = np.array([], dtype=np.float32)
       count = 0
       temp = 0
       for j in range(len(self.CCFdata)):
           if lowerfreq <= self.CCFdata[j][0] and upperfreq > self.CCFdata[j][0]:
               temp = self.CCFdata[j][1] 
               temp_spec = np.append(temp_spec, temp)
       return temp_spec.astype('float32')  
   
    def cal_GainCircuit(self, freq): 
       upperfreq = freq + AcqBW/2 
       lowerfreq = freq - AcqBW/2
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
   
    def trim_spectrum(self,spectrum):
        final_sample= 373852
        specsize=len(spectrum[:,0])
        
        #_log.info("Usable number of Samples: %i "%(final_sample))
        starttrim = int((specsize-final_sample)/2) 
        stoptrim = int(specsize-(specsize-final_sample)/2)
        freq = np.array(spectrum[:,0])
        fft= np.array(spectrum[:,1])
        x = freq[starttrim:stoptrim],fft[starttrim:stoptrim]
        spec = np.array(x, dtype=np.float32)
        return spec

    def to_decibels(self,x):
        calfactor = 1000/50/(523392/2)                    
        return 10*np.log10(x*x*calfactor)
    
    def change_freq_channel(self, spectrum, factor):
        outputChannel = int(len(spectrum[:,0])/factor)
        outputfreqlist = np.zeros(outputChannel)
        outputspeclist = np.zeros(outputChannel)
        for i in range(outputChannel):
            outputfreqlist[i] = np.mean(spectrum[i*factor:(i*factor+factor),0])
            outputspeclist[i] = np.mean(spectrum[i*factor:(i*factor+factor),1])
        x = outputfreqlist, outputspeclist
        y  = np.array(x, dtype=np.float32)
        return y

    def readIQDataBin_Trimmed(self, path, filename, start_freq, stop_freq, array_size = 40, resfact=1):
        # 40 MHz kom hier in
        temp_spec = np.array([])
        temp_freq = np.array([])
        arraydata =np.load(path + filename)
        spectrum = arraydata.T
        #spect_trim = spectrum[:,0],spectrum[:,1]
        spectrim = self.trim_spectrum(spectrum)
        trimspec = np.array(self.change_freq_channel(spectrim.T,resfact), dtype=np.float32).T
        x = int(len(trimspec)/array_size)
        for i in range(int(array_size)):
        #trimspec = np.array(self.change_freq_channel(self.trim_spectrum(spectrum),resfact))
            temp = np.mean(trimspec[i*x:x*(i+1),1])
            tempfreq = np.mean(trimspec[i*x:x*(i+1),0])
            temp_spec = np.append(temp_spec, temp)
            temp_freq = np.append(temp_freq, tempfreq)
        spec = self.to_decibels(temp_spec)
        f = np.linspace(start_freq,stop_freq,array_size)
        f.astype('float32') 
        temp = temp_freq, spec
        y = np.array(temp, dtype=np.float32)
        return y
    
    # genertes a list of standart center frequencies        
    def generate_CFlist(self,startfreq, stopfreq):
        lowerlimit = 80e6 #  Hz Defines the lower frequency limit
        upperlimit = 6000e6 # Hz Defines the upper freqzency limit
        if startfreq > stopfreq: 
            temp = startfreq
            startfreq = stopfreq
            stopfreq = temp
        if lowerlimit >= startfreq and startfreq <= upperlimit: startfreq = lowerlimit
        if lowerlimit >= stopfreq and stopfreq <= upperlimit: stopfreq = upperlimit
        if startfreq > stopfreq: 
            temp = startfreq
            startfreq = stopfreq
            stopfreq = temp
            
        cfreq_tabelle = list(range(int(round((stopfreq/1e6-80)/40+0.499999))-int((startfreq/1e6-80)/40)))
        for i in range(len(cfreq_tabelle)):
            cfreq_tabelle[i]= ((i+int((startfreq/1e6-80)/40))*40+100)*1e6
        return cfreq_tabelle
    
    
    def calCCF(self,spectrum, CCF, r, Lcable, G_LNA, antennaEfficiency): # returns in [dBuV/m]
         # spectrum numpy array
         #get the 1MHz average of the spectrum 
         temp = 0
         temp = -G_LNA - Lcable - (10.0 * np.log10(antennaEfficiency)) - CCF #+ (10.0 * np.log10(Z0 / (4.0 * np.pi * (r*r)))) + 90.0
         #print('temp_from_calCCf = %s \n'% temp)
         y = spectrum[0], spectrum[1]+temp
         y = np.array(y, dtype=np.float32)
         return y
               
    def cal_data(self,entry):
       tstart = time.time()
       resfact = 1
       y = "Power [dBm]" #"Electrical Field Strength [dBuV/m]"
       resolution =  0.1069943751528491*resfact
       temp_spec = np.array([])
       temp_freq = np.array([])
           #print('Invalid directory defined')
       path, filename, Start_freq, Stop_freq, Lcable, antennaEfficiency = self.fetch(entry)
       Freqlist =self.generate_CFlist(int(Start_freq)*1e6,int(Stop_freq)*1e6) 
       
           # Extract the CCF for the correct freqeuncy range and get the raw spectrum data in one array 
       for i in Freqlist: # i = 40 Mhz incrementations
           CCF = self.cal_CCF(i) # CCF will output 40 points
           G_LNA = self.cal_GainCircuit(i)
           upperfreq = (i + AcqBW/2)
           lowerfreq = (i - AcqBW/2) 
           print(i)
           filename = filename+str(int(i/1e6))+"MHz.npy"
           x = np.array(self.readIQDataBin_Trimmed(path,filename, lowerfreq, upperfreq,40), dtype=np.float32)
           print(int(Lcable))
           Spec = self.calCCF(x, CCF, r, int(Lcable), G_LNA, float(antennaEfficiency))
           tempfreq = Spec[:,0]/1e6
           temp = Spec[:,1]
           temp_spec = np.append(temp_spec, temp)
           temp_freq = np.append(temp_freq, tempfreq)
           #self.fig_plot.plot(Spec[0]/1e6,Spec[1], c=color[1])
       print('Path: %s \nFilename: %s \nStart Frequency (MHz): %s \nStop Frequency (MHz): %s \nLNA gain (dB): %s\nCable losses (dB): %s \nAntenna Efficiency: %s \n' % (path, filename, lowerfreq, upperfreq, G_LNA, Lcable, antennaEfficiency)) 
       print(len(temp_spec))
       self.fig_plot.plot(temp_freq,temp_spec, c=color[1])
       self.fig_plot.set_ylabel(y)
       self.fig_plot.set_xlabel("Frequency (MHz) (resolution %.3f kHz)"%resolution)
#      print('Path: %s \nFilename: %s \nStart Frequency (MHz): %s \nStop Frequency (MHz): %s \nLNA gain (dB): %s\nCable losses (dB): %s \nAntenna Efficiency: %s \n' % (self.Path, self.Filename, self.Start_freq, self.Stop_freq, self.G_LNA, self.Lcable, self.antennaEfficiency)) 
       self.canvas.draw()

       print ('time:' , (time.time()-tstart))
if __name__ == '__main__':
   window = Tk()
   start = GUI_set_up(window, fields)
   window.mainloop() 
