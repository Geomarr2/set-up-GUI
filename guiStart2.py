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
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import dialog as Dialog
from tkinter import filedialog as LoadFileDialog
from tkinter import scrolledtext as ScrolledText

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
class FilenameEntry(Frame):

	def __init__(self, master, text):
		Frame.__init__(self, master)
		Label(self, text=text).pack(side=LEFT)
		self.filename = StringVar()
		Entry(self, textvariable=self.filename).pack(side=LEFT, fill=X)
		Button(self, text="Browse...", command=self.browse).pack(side=RIGHT)

	def browse(self):
		file = LoadFileDialog(self).go(pattern='*')
		if file:
			self.filename.set(file)

	def get(self):
		return self.filename.get()

class ButtonBar(Frame):

	def __init__(self, master, left_button_list, right_button_list):
		Frame.__init__(self, master, bd=2, relief=SUNKEN)
		for button, action in left_button_list:
			Button(self, text=button, command=action).pack(side=LEFT)
		for button, action in right_button_list:
			Button(self, text=button, command=action).pack(side=RIGHT)

class FileNotFoundMessage(Dialog):

	def __init__(self, master, filename):
		Dialog.__init__(self, master, title = 'File not found',
			text = 'File ' + filename + ' does not exist',
			bitmap = 'warning', default = 0,
			strings = ('Cancel',))

class GUI_set_up:
    def __init__(self,  window, fields):
        self.Path= "D:/Geomarr/Spectrum/"
        self.Filename = "0RFISpectrum"
        self.Start_freq = 1000
        self.Stop_freq = 1040
        self.G_LNA = 10 
        self.Lcable = -1
        self.antennaEfficiency = 0.75
        
        self.window = window
        self.window.title("RFI Chamber")
       #self.frame = Frame(window)
        #self.entry = Entry(self.frame)
        self.entry_data = self.makeform(fields)
        self.window.bind('<Return>', (lambda event,e=self.entry_data: self.fetch()))
        #self.label =  Label(self.frame, width=22, text=field+": ", anchor='w')
        self.button_get = Button(self.window,text = 'accept', command = (lambda e=self.entry_data: self.fetch()))
        self.button_get.pack(side=LEFT, padx=5, pady=5)
        self.button_plot=Button(window,text = 'show plot', command=(lambda e=self.entry_data: self.cal_data()))
        self.button_plot.pack(side=LEFT, padx=5, pady=5)
        
        
    def fetch(self):          
        self.Path= self.entry_data['Path'].get()
        self.Filename = self.entry_data['Filename'].get()
        self.Start_freq = self.entry_data['Start Frequency (MHz)'].get()
        self.Stop_freq = self.entry_data['Stop Frequency (MHz)'].get()
        self.G_LNA = self.entry_data['LNA gain (dB)'].get() 
        self.Lcable = self.entry_data['Cable losses (dB)'].get()
        self.antennaEfficiency = self.entry_data['Antenna efficiency'].get()
        print('Path: %s \nFilename: %s \nStart Frequency (MHz): %s \nStop Frequency (MHz): %s \nLNA gain (dB): %s\nCable losses (dB): %s \nAntenna Efficiency: %s \n' % (self.Path, self.Filename, self.Start_freq, self.Stop_freq, self.G_LNA, self.Lcable, self.antennaEfficiency)) 
       
    def get_data(l):
        l.append(self.entry.get())
        print(l)
        
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
       
    def cal_data(self):
       F = []
       tempCCFData = []
       tempSpec = []
       if len(self.Path) == 0:
           print('Invalid directory defined')
       else:
           start_freq = int(self.Start_freq)*1e6
           stop_freq = int(self.Stop_freq)*1e6
           Freqlist =cal.generate_CFlist(start_freq,stop_freq)
           Length_freq = len(Freqlist)

           # Average CCF for each given centre frequency and AcqBW
           #upperfreq = start_freq + AcqBW/2 
           #lowerfreq = stop_freq - AcqBW/2
        # get the spectrum
           G_LNA = int(self.G_LNA) 
           Lcable = int(self.Lcable)
           antennaEfficiency = float(self.antennaEfficiency) 
           path = self.Path +"_gLNA"+str(G_LNA)+"_Lcable"+str(Lcable)+"_EffAnt"+str(antennaEfficiency)+"/"
           path = self.Path
           filename = self.Filename  
        
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
           fig = Figure(figsize=(10,10))
           fig_plot = fig.add_subplot(111)
           resfact = 1
           for i in range(Length_freq):
               trimspec = np.array(cal.change_freq_channel(cal.trim_spectrum(tempSpec[i,:,:]),resfact))
                #trimspec = np.array(trim_spectrum(spectrum))
               spec = cal.to_decibels(trimspec[1])
               resolution =  0.1069943751528491*resfact
               fig_plot.plot(trimspec[0]/1e6,spec, c=color[i])
                #fig_plot.ylim(-80,0)
               fig_plot.set_ylabel("Power (dBm)")
               fig_plot.set_xlabel("Frequency (MHz) (resolution %.3f kHz)"%resolution)
               #cal.plot_stiched_spectrum(tempSpec[i,:,:],color[i])   
               
           canvas = FigureCanvasTkAgg(fig, master=self.window)
           canvas.get_tk_widget().pack(side=BOTTOM, fill=X, padx=5, pady=5)
           canvas.draw()
           
if __name__ == '__main__':
   window = Tk()
   start = GUI_set_up(window, fields)

                         # The lambda function used here takes one argument, 
                                                                                # and returns None

  # root.bind('<Return>',  cal.plot_stiched_spectrum(Spec,color[1]))
   window.mainloop() 
   
