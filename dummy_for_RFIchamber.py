# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:15:23 2018
@author: G-man
"""
try:
    # Python2
    from Tkinter import * 
except ImportError:
    # Python3
    from tkinter import *
import matplotlib
#matplotlib.use('cairo')
#from IPython import get_ipython
#get_ipython().magic('reset -sf')
import numpy as np
from io import StringIO
global fields 
#import calData as cal
import os
import csv
import pandas as pd
import sys
import time
from tkinter import *
from scipy.interpolate import spline
from tkinter import messagebox as tkMessageBox
from tkinter import filedialog as tkFileDialog
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.widgets import Lasso
from PIL import ImageTk,Image
import re
import mpifrRFIHeader
#import matplotlib as plt

from matplotlib.widgets import Lasso
from matplotlib.figure import Figure
#import RFIHeaderFile
#from tkinter import Label, Button, Radiobutton, IntVar
colorbg= "#d469a3"
FileCCF="CCF4.csv"
PATHCCF = "D:/Testskripte/RFcable/CCF4.csv"
PATHGAINCIRCUIT = "D:/Testskripte/RFcable/GainCircuitNew.csv"
FileGainCircuit = "GainCircuit.csv"
fields = 'Path', 'Filename', 'Start Frequency (MHz)', 'Stop Frequency (MHz)', 'LNA gain (dB)', 'Cable losses (dB)','Antenna efficiency'
fields_zoom = "Start Frequency (MHz): ", "Stop Frequency (MHz): ","Maximum amplitude: ","Minimum amplitude: "
color_data_set1 = ['lightskyblue','lightcoral','m','y']
color_data_set2 = ['red','limegreen','c','plum','goldenrod']
#global org_data 
AcqBW = 40e6
bandwidth = AcqBW #Hz
#StartFreq = 1000e6
#StopFreq = 1200e6

#Lcable = -1  #dB cable losses
#antennaEfficiency = 0.75 
Nsample = 523852
r = 1
CCFtemp = []
Gaintemp = []
test = []
# TEST path DataPath = "D:/Geomarr/Spectrum/"       #Path to save the spectra
#TEST filename = 0RFISpectrum
#TEST start_freq = 1000
#TEST stop_freq = 1040

#---------Plotting constants------#
DEFAULT_XAXIS   = "P_bary (ms)"
DEFAULT_YAXIS   = "Sigma"
BESTPROF_DTYPE  = [
    ("Text files","*.txt"),("all files","*.*")
    ]
PLOTABLE_FIELDS = [key for key,dtype in BESTPROF_DTYPE if dtype=="float32"]
PLOT_SIZE = (8,5)
MPL_STYLE = {
    "text.color":"lightblue",
    "axes.labelcolor":"lightblue",
    "axes.edgecolor":"black",
    "axes.facecolor":"0.4",
    "xtick.color": "lightblue",
    "ytick.color": "lightblue",
    "figure.facecolor":'black',
    "figure.edgecolor":"black",
    "text.usetex":False
}
mpl.rcParams.update(MPL_STYLE)

#-----------Misc.-------------#
DEFAULT_DIRECTORY = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/RFIDummyDatasetGUI/"#os.getcwd()

#----------Style options---------#
DEFAULT_PALETTE = {"foreground":"blue","background":"grey"}
DEFAULT_PALETTE_NEW = {"foreground":"blue","background":"gray90"}
DEFAULT_STYLE_1 = {"foreground":"black","background":"lightblue"}
DEFAULT_STYLE_2 = {"foreground":"gray90","background":"darkgreen"}
DEFAULT_STYLE_3 = {"foreground":"gray90","background":"darkred"}
DEFAULT_STYLE_NEW = {"foreground":"gray90","background":"darkgreen"}
class NavSelectToolbar(NavigationToolbar2TkAgg): 
    def __init__(self, canvas,root,parent):
        self.canvas = canvas
        self.root   = root
        self.parent = parent
        NavigationToolbar2TkAgg.__init__(self, canvas,root)
        #self._zoomrect_default=NavigationToolbar2TkAgg.zoom
        
        self.lasso_button = self._custom_button(text="lasso",command=lambda: self.lasso(
                lambda inds: self.parent.multi_select_callback(inds),"lasso"),**DEFAULT_STYLE_1)
        self.pick_button = self._custom_button(text="select",command=lambda: self.picker(
                lambda ind: self.parent.single_select_callback(ind),"select"),**DEFAULT_STYLE_1)
        
    def _custom_button(self, text, command, **kwargs):
        button = Button(master=self, text=text, padx=2, pady=2, command=command, **kwargs)
        button.pack(side=LEFT,fill="y")
        return button


class GUI_set_up:
    def __init__(self, root):     
        self.root = root
        self.fields = fields
        self.x_factor = 20
        self.root.title("RFI Chamber")
        self.top_frame = Frame(self.root)
        self.top_frame.pack(side=TOP)
        self.bottom_frame = Frame(self.root)
        self.bottom_frame.pack(side=BOTTOM)
        
        #self.text_frame = Frame(self.root)
        #self.text_frame.pack()
        
        self.top_frame_plot = Frame(self.root)
        self.top_frame_plot.pack(side=LEFT)
        self.plot_frame_toolbar = Frame(self.top_frame_plot)
        self.plot_frame_toolbar.pack(side=TOP)
        self.plot_frame = Frame(self.top_frame_plot)
        self.plot_frame.pack(side=TOP)
        
        self.options_frame = Frame(self.top_frame) 
        self.options_frame.pack(side=LEFT)

        
        self.input_frame = Frame(self.top_frame)
        self.input_frame.pack(side=TOP)
        self.buttons_frame = Frame(self.top_frame_plot)
        self.buttons_frame.pack(side=RIGHT)
        self.dir_button_frame = Frame(self.bottom_frame)
        self.dir_button_frame.pack(side=RIGHT)
        self.newRBW_button_frame = Frame(self.top_frame)
        self.newRBW_button_frame.pack(side=BOTTOM)
        
        self.options    = GUIOptions(self.options_frame,self)
        self.options.plot()

        
class GUIOptions(object):
    def __init__(self,root,parent):
        self.root = root
        self.parent = parent
        self.Path= "C:/Users/geomarr/Documents/GitHub/set-up-GUI/RFIDummyDataset/"
        self.PathTEST= "C:/Users/geomarr/Documents/GitHub/set-up-GUI/RFIDummyDataset/"
        self.Filename = "Spectrum_"
        self.CenterFrequency = 0.
        self.Nchannels = 373851
        self.G_LNA = 0. 
        self.Lcable = 0.
        self.antennaEfficiency = 0.
        self.data = None
        self.scaling_factor=0.           # number of points full data set
        
        # set counters
        self.missingData = 0
        self.UserInputMissingData = 0
        self.max_plot = []
        self.min_plot = []
        
        #create headerfile
        self.head = mpifrRFIHeader.HeaderInformation()
        self.headerFile = []
        self.dataFile = []
        #self.Start_freq = 1000*1e6
        #self.Stop_freq = 2000*1e6
        self.Bandwidth = 40*1e6
        self.headerInfo = []
        self.headerFile2 = []
        self.dataFile2 = []
        #self.Start_freq2 = 1000*1e6
        #self.Stop_freq2 = 2000*1e6
        self.Bandwidth2 = 40*1e6
        self.headerInfo2 = []
        self.CFreq = 0
        self.CFreq2 = 0
        
        self.zoom_data = []
        self.original_data = []
        self.original_freq = []
        self.zoom_data2 = []
        self.original_data2 = []
        self.original_freq2 = []
        self.plotOne = False
        self.plotTwo = False
        
        self.zoom_Start_freq=0
        self.zoom_Stop_freq=0
        self.zoom_max=0
        self.zoom_min=0
        self.zoom_trigger1 = False
        self.zoom_trigger = False
        self.zoom_trigger2 = False
        self.original = False
        
        self.mode_select_frame = Frame(self.root,pady=20)
        self.mode_select_frame.pack(side=TOP,fill=X,expand=1)
        self.ax_opts_frame = Frame(self.root)
        self.ax_opts_frame.pack(side=TOP,expand=1)
        self.view_toggle_frame = Frame(self.root,pady=20)
        self.view_toggle_frame.pack(side=TOP,fill=X,expand=1)
        
        #self.entry_data = self.makeform()
        self.newRWB_entry_data = self.new_res_BW()
        self.canvas = None
        self.figure = None
        self.fig_plot = None
        self.toolbar = None
        
        
        self.LNApath = PATHGAINCIRCUIT
        self.CCFpath = PATHCCF
        
        
        # button to get new BW of the display resolution 
        
        self.newRBW_opts_frame = Frame(self.parent.newRBW_button_frame,pady=1)
        
        self.newRBW_opts_frame.pack(side=RIGHT,fill=BOTH,expand=1)
        self.show_newRBW_button=self._custom_button(
                self.newRBW_opts_frame,text = 'Replot Second Data', 
                command=(lambda self=self:self.zoomActivate2Data()),
                **DEFAULT_STYLE_2)
        
        self.newRBW_opts_frame = Frame(self.parent.newRBW_button_frame,pady=1)
        
        self.newRBW_opts_frame.pack(side=RIGHT,fill=BOTH,expand=1)
        self.show_newRBW_button=self._custom_button(
                self.newRBW_opts_frame,text = 'Replot', 
                command=(lambda self=self:self.zoomActivate()),
                **DEFAULT_STYLE_2)
        
        
        self.clear_opts_frame = Frame(self.parent.buttons_frame,pady=1)
        self.clear_opts_frame.pack(side=TOP,fill=X,expand=1)
        self.clear_button=self._custom_button(
                self.clear_opts_frame,text = 'Clear Plot', 
                command=(lambda self=self: self.clear_data()),
                **DEFAULT_STYLE_2)
        
        
        self.misc_opts_frame = Frame(self.parent.buttons_frame,pady=1)
        self.misc_opts_frame.pack(side=TOP,fill=X,expand=1)
        self.quit_button = self._custom_button(
        self.misc_opts_frame,"Quit",
        self.quit,**DEFAULT_STYLE_3)
        
        self.top_frame = Frame(self.parent.dir_button_frame)
        self.top_frame.pack(side=TOP,anchor=W)
        
        self.bottom_frame = Frame(self.parent.dir_button_frame)
        self.bottom_frame.pack(side=BOTTOM,fill=BOTH,expand=1)
        Label(self.top_frame,text="Directory:",padx=8,pady=2,height=1).pack(side=LEFT,anchor=W)
        self.directory_entry = Entry(self.top_frame,width=90,bg="lightblue",
                                    fg="black",highlightcolor="lightblue",insertbackground="black",
                                    highlightthickness=2)
        
        self.directory_entry.pack(side=LEFT,fill=BOTH,expand=1,anchor=W)
        self.directory_entry.insert(0,DEFAULT_DIRECTORY)
        Button(self.top_frame,text="Browse",command=self.launch_dir_finder,**DEFAULT_STYLE_1
                  ).pack(side=LEFT,fill=BOTH,expand=1,anchor=W)
        self.submit_button = Button(self.bottom_frame,text="Load Data",width=60,
                                       command=lambda self=self:self.dump_filenames(),
                                       **DEFAULT_STYLE_2).pack(side=LEFT,anchor=SW)
        

        self.submit_button = Button(self.bottom_frame,text="Load Second Data",width=60,
                                       command=lambda self=self:self.dump_filenames_data2(),
                                       **DEFAULT_STYLE_2).pack(side=RIGHT,anchor=SE)
        
    def printHeaderInfo(self,dataFile,headerFile):
#Load in all the header file get the info and the BW from max and min center freq
        
        file = [value for counter, value in enumerate(headerFile)]
        headerInfo = [open(headerFile[count],'r').readlines() for count, val in enumerate(headerFile)]
        temp = [value[Cnt+1] for counter, value in enumerate(headerInfo) for Cnt, Val in enumerate(value) if Val == 'Center Frequency in Hz:\n']
        temp = [float(value[:-1]) for counter, value in enumerate(temp) if value.endswith('\n')]
        temp = np.array(temp)#.astype('float32')
        
        
        tempID = [value[Cnt+1] for counter, value in enumerate(headerInfo) for Cnt, Val in enumerate(value) if Val == 'Unique Scan ID:\n']
        tempID = [value[:-1] for counter, value in enumerate(tempID) if value.endswith('\n')]
        
#check if all the numpy arrays is there
        if len(temp) == len(dataFile):
            #temp = [np.load(value) for counter, value in enumerate(self.dataFile) for counterHeader, valueHeader in enumerate(temp) if value == (valueHeader+'.npy')]
            file2 = open(headerFile[0], "r")
            msg = []
            headerInfo = file2.readlines()
            if headerInfo[6] != 'default\n' and headerInfo[8] != 'default\n' and headerInfo[18] != 'default\n' and headerInfo[20] != 'default\n'and headerInfo[22] != 'default\n'and headerInfo[24] != 'default\n':
                self.Bandwidth = float(headerInfo[6])
                self.CCFpath = headerInfo[18]
                self.antennaEfficiency = float(headerInfo[20])
                self.Lcable = float(headerInfo[22])
                self.LNApath = headerInfo[24]
                self.head.nSample = headerInfo[8]
            else:
                self.Bandwidth = 40*1e6
                self.LNApath = PATHGAINCIRCUIT
                self.CCFpath = PATHCCF
                self.antennaEfficiency = 0.75
                self.Lcable = 1
                self.head.nSample = 373851
            for x in headerInfo:
                msg.append(x)
            file2.close()
            tkMessageBox.showinfo(title = "Acquisition information", message = msg)
        else:
            tkMessageBox.showwarning('Warning','Data is missing')
        return msg, temp, tempID



    def printHeaderInfoZOOM(self,dataFile,headerFile,cfreq):
#Load in all the header file get the info and the BW from max and min center freq
        
        file = [value for counter, value in enumerate(headerFile)]
        headerInfo = [open(headerFile[count],'r').readlines() for count, val in enumerate(headerFile)]
        
        temp = [value[Cnt+1] for counter, value in enumerate(headerInfo) for Cnt, Val in enumerate(value) if Val == 'Center Frequency in Hz:\n']
        temp = [float(value[:-1]) for counter, value in enumerate(temp) if value.endswith('\n')]
        temp = np.array(temp).astype('float32')
        
        tempID = [value[Cnt+1] for counter, value in enumerate(headerInfo) for Cnt, Val in enumerate(value) if Val == 'Unique Scan ID:\n']
        tempID = [value[:-1] for counter, value in enumerate(tempID) if value.endswith('\n')]
        
        s = [dataFile[count_cfreq] for count_cfreq, value_cfreq in enumerate(temp) if value_cfreq >= (cfreq-20*1e6) and value_cfreq <= (cfreq+20*1e6)]
        return s[0]
    
    def askUserMultipleQuestion(self,prompt, fields):
        self.root2 = Toplevel(self.root,bg="gray90")
        #root2.title("Continue with missing data set")
        label_frame = Frame(self.root2)
        label_frame.pack(side=TOP,fill=BOTH,expand=1)
        button_frame = Frame(self.root2)
        button_frame.pack(side=BOTTOM,fill=BOTH,expand=1)

        Label(label_frame, text=prompt).pack()
        Label(label_frame, text=fields[0]).pack()
        Label(label_frame, text=fields[1]).pack()
        Label(label_frame, text=fields[2]).pack()
        v = Entry(button_frame)
        v.pack()
        
        button_submit_frame = Frame(button_frame)
        button_submit_frame.pack(side=BOTTOM,fill=BOTH,expand=1)
        Button(button_submit_frame,text="Submit", command=lambda v=v:self.callback(v),**DEFAULT_STYLE_NEW).pack()
        Button(button_submit_frame,text="Done", command=self.root2.destroy,**DEFAULT_STYLE_NEW).pack()

    
    def callback(self,v):
        self.UserInputMissingData = int(v.get())
        print ("User's response was: %d"%self.UserInputMissingData)
        
    def nextCommand(self):
        tkMessageBox.ABORT
        self.missingData = 1
        return 
    
    def CCF_DatafileCSV(self,path):
        data = []
        if path.endswith('\n'):
            path = path[:-1]
        f = open(path, "r")
        row = f.readlines()
        f.close()
        splitRow = [row[i].split(',') for i in range(len(row))]
        data = [value for counter, value in enumerate(splitRow)]
        data =np.array(data, dtype = 'float32')    
        return data       
    
    def get_CCF(self,scaling_factor, Stop_freq, Start_freq): 
       CCFdata = self.CCF_DatafileCSV(self.CCFpath)
       temp_spec = np.array([],dtype='float32')
       CCFTemp = [CCFdata[cnt,1] for cnt, val in enumerate(CCFdata[:,0]) if val >= Start_freq and val <= Stop_freq]
       CCFFreq = [CCFdata[cnt,0] for cnt, val in enumerate(CCFdata[:,0]) if CCFdata[cnt,0] >= Start_freq and CCFdata[cnt,0] <= Stop_freq]
       
       x = np.linspace(Start_freq, Stop_freq, scaling_factor)
       temp_spec = np.interp(x, CCFFreq, CCFTemp)
       temp_spec2 = [x, temp_spec]
       CCFtemp.append(temp_spec2)
       return temp_spec
   
    def GainLNA_DatafileCSV(self,path):
        data = np.array([])  
        temp_data = np.array([]) 
        if path.endswith('\n'):
            path = path[:-1]
        f = open(path, "r")
        row = f.readlines()
        splitRow = [row[i].split(';') for i in range(len(row))]
        data = [value for counter, value in enumerate(splitRow)]#(np.append(data, row) for row in csvdata)
        
        newData = [data[i] for i in range(len(data)) if 2 <= len(data[i])]
        newDataTemp = np.array([newData[i] for i in range(1,len(newData))])
        freq = newDataTemp[:,0]
        dataG = (newDataTemp[:,1])
        freq = [float(i.replace(',','.')) for i in freq]
        dataG = [float(i.replace(',','.')) for i in dataG]
        temp = [freq,dataG]
        GainLNA = np.array(temp).astype('float32')
        return GainLNA
        
    def loadDataFromFile(self,Cfreq,dataFile):
        spec = np.load(dataFile)
        freq = np.linspace(Cfreq-20*1e6,Cfreq+20*1e6,len(spec))
        temp = [freq,spec]
        
        return np.array(temp, dtype=np.float32) 

    
    def dump_filenames(self):
#Load in vereything and seperate header and data file
        if self.zoom_trigger1 == True or self.zoom_trigger2 == True:
            self.clear_plot()
            self.zoom_trigger1 = False
            self.zoom_trigger2 = False
        path = None
        new_path = None
        self.plotOne = True
        dataFile = []
        headerFile = []
        path = self.directory_entry.get()
        path.replace("\\","/") + "/"
        new_path = path.split(' ')
        for i in range(len(new_path)): 
            if new_path[i].endswith(".rfi"):
                headerFile.append(new_path[i])
            elif new_path[i].endswith(".npy"):
                dataFile.append(new_path[i])
        msg,tem, tempID = self.printHeaderInfo(dataFile,headerFile)
        self.CFreq = tem
        self.Bandwidth = float(msg[6])
        Start_freq = min(tem) - 20*1e6
        Stop_freq = max(tem) + 20*1e6
        self.xlim_Start_freq = Start_freq/1e6
        self.xlim_Stop_freq = Stop_freq/1e6
        
        self.CenterFrequency = (Stop_freq-Start_freq)/2
        self.dataFile = dataFile
        self.headerFile = headerFile

        dataFile = [dataFile[i] for i in range(len(dataFile)) if tem[i] <= 6000*1e6 and tem[i] >= 100*1e6]
        te = [tem[i] for i in range(len(tem)) if tem[i] <= 6000*1e6 and tem[i] >= 100*1e6]
        Start_freq = min(te) - 20*1e6
        Stop_freq = max(te) + 20*1e6
        self.Start_freq = Start_freq
        self.Stop_freq = Stop_freq 
        scaling_factor = 40
        for cnt, value in enumerate(te):
            Startfreq = 0
            Stopfreq = 0
            Startfreq = value - (20*(1e6))
            Stopfreq = value + (20*1e6)
            #data = self.head.readFromFile(headerFile[cnt])
            original_data = self.loadDataFromFile(value,dataFile[cnt])
            self.calibrateData(self.read_reduce_Data(original_data,scaling_factor), Stopfreq, Startfreq, color_data_set1, scaling_factor)

        self.fig_plot.set_ylim(-50, 100)
        self.fig_plot.set_xlim(self.xlim_Start_freq,self.xlim_Stop_freq)
        self.fig_plot.legend(self.leg_data,self.leg)
        self.canvas.draw()   
        
    def dump_filenames_data2(self):
#Load in vereything and seperate header and data file
        if self.zoom_trigger1 == True or self.zoom_trigger2 == True:
            self.clear_plot()
            self.zoom_trigger1 = False
            self.zoom_trigger2 = False
        self.plotTwo = True
        path = None
        new_path = None
        dataFile = []
        headerFile = []
        path = self.directory_entry.get()
        path.replace("\\","/") + "/"
        new_path = path.split(' ')
        for i in range(len(new_path)): 
            if new_path[i].endswith(".rfi"):
                headerFile.append(new_path[i])
            elif new_path[i].endswith(".npy"):
                dataFile.append(new_path[i])
        msg,tem, tempID = self.printHeaderInfo(dataFile,headerFile)
        self.CFreq2 = tem
        self.Bandwidth = float(msg[6])
        self.headerFile2 = headerFile
        self.dataFile2 = dataFile
               
        dataFile = [dataFile[i] for i in range(len(dataFile)) if tem[i] <= 6000*1e6 and tem[i] >= 100*1e6]
        te = [tem[i] for i in range(len(tem)) if tem[i] <= 6000*1e6 and tem[i] >= 100*1e6]
        Start_freq = min(te) - 20*1e6
        Stop_freq = max(te) + 20*1e6
        self.Start_freq2 = Start_freq
        self.Stop_freq2 = Stop_freq
        scaling_factor = 40
        for cnt, value in enumerate(te):
            Startfreq = 0
            Stopfreq = 0
            Startfreq = value - (20*(1e6))
            Stopfreq = value + (20*1e6)
            original_data = self.loadDataFromFile(value,dataFile[cnt])
            self.calibrateData(self.read_reduce_Data(original_data,scaling_factor), Stopfreq, Startfreq, color_data_set2, scaling_factor)
            
        self.xlim_Start_freq = Start_freq/1e6
        self.xlim_Stop_freq = Stop_freq/1e6            
        self.fig_plot.set_ylim(-50,100)
        self.fig_plot.set_xlim(self.xlim_Start_freq,self.xlim_Stop_freq)
        self.fig_plot.legend(self.leg_data,self.leg)
        self.canvas.draw()           
        
    def zoom_dump_data(self, zoom_Start_freq, zoom_Stop_freq, scaling_factor,color, dataFile, CFFreq,Start_freq, Stop_freq):
        # original data start frequency and stop frequency and datafile
#        Start_freq = 0
 #       Stop_freq = 0
#        dataFile = []
        zoom_Start_freq_plot = zoom_Start_freq - (40*1e6)
        zoom_Stop_freq_plot = zoom_Stop_freq + (40*1e6)
#        if self.plotOne:
            
 #           Start_freq = self.Start_freq 
  #          Stop_freq = self.Stop_freq    

            
   #     elif self.plotTwo:
    #        Start_freq = self.Start_freq2 
     #       Stop_freq = self.Stop_freq2    
      #      dataFile = self.dataFile2 
       #     temp = self.CFreq2
        
#        if Start_freq < 100*1e6 and Stop_freq > 6000*1e6:
            
 #           Start_freq = 100*1e6
  #          Stop_freq = 6000*1e6
   #         dataFile = [dataFile[i] for i in range(len(dataFile)) if temp[i] <= 6000*1e6 and temp[i] >= 100*1e6]
    #        temp = [temp[i] for i in range(len(temp)) if temp[i] <= 6000*1e6 and temp[i] >= 100*1e6]
        
 #       elif Stop_freq > 6000*1e6:
            #dataFile = dataFile[0:len(dataFile)-5]
  #          dataFile = [dataFile[i] for i in range(len(dataFile)) if temp[i] <= 6000*1e6]
   #         Stop_freq = 6000*1e6
    #        temp = [temp[i] for i in range(len(temp)) if temp[i] <= 6000*1e6]
            
     #   elif Start_freq < 100*1e6:
            #dataFile = dataFile[1:len(dataFile)]
      #      Start_freq = 100*1e6
       #     dataFile = [dataFile[i] for i in range(len(dataFile)) if temp[i] >= 100*1e6]
        #    temp = [temp[i] for i in range(len(temp)) if temp[i] >= 100*1e6]
        
        dataFile = [dataFile[i] for i in range(len(dataFile)) if CFFreq[i] <= 6000*1e6 and CFFreq[i] >= 100*1e6]
        te = [CFFreq[i] for i in range(len(CFFreq)) if CFFreq[i] <= 6000*1e6 and CFFreq[i] >= 100*1e6] 
              
        for cnt, value in enumerate(te):
             if value >= zoom_Start_freq_plot and value <= zoom_Stop_freq_plot:
                 Startfreq = value - (20*1e6)
                 Stopfreq = value + (20*1e6)
                 originaldata = self.loadDataFromFile(value,dataFile[cnt])
                 self.calibrateData(self.read_reduce_Data(originaldata, scaling_factor), Stopfreq, Startfreq, color, scaling_factor)

        self.fig_plot.set_ylim(self.zoom_min,self.zoom_max)
        self.fig_plot.set_xlim(self.zoom_Start_freq/1e6,self.zoom_Stop_freq/1e6)
        self.fig_plot.legend(self.leg_data,self.leg)
        self.canvas.draw()   
        
    def zoom_dump_data_org(self, zoom_Start_freq, zoom_Stop_freq, color,headerFile, dataFile, CFFreq,Start_freq, Stop_freq):
        # original data start frequency and stop frequency and datafile
#        Start_freq = 0
#        Stop_freq = 0
        scaling_factor = int(self.head.nSample)
#        dataFile = []
        original_data = []# np.array([], dtype='float32')
        d = []
        counter = []
        # I want to over sample
        zoom_Start_freq_plot = zoom_Start_freq - (40*1e6)
        zoom_Stop_freq_plot = zoom_Stop_freq + (40*1e6)

    #    if self.plotOne:
            
     #       Start_freq = self.Start_freq 
      #      Stop_freq = self.Stop_freq
       #     dataFile = self.dataFile
        #    headerFile = self.headerFile
         #   tempfreq = self.CFreq
#        elif self.plotTwo:
 #           dataFile = self.dataFile2
  #          headerFile = self.headerFile2
   #         Start_freq = self.Start_freq2 
    #        Stop_freq = self.Stop_freq2    
     #       tempfreq = self.CFreq2
        
        if Start_freq < 100*1e6 and Stop_freq > 6000*1e6:
            
            Start_freq = 100*1e6
            Stop_freq = 6000*1e6
            dataFile = [dataFile[i] for i in range(len(dataFile)) if CFFreq[i] <= 6000*1e6 and CFFreq[i] >= 100*1e6]
            CFFreq = [CFFreq[i] for i in range(len(CFFreq)) if CFFreq[i] <= 6000*1e6 and CFFreq[i] >= 100*1e6]
        
        elif Stop_freq > 6000*1e6:
            #dataFile = dataFile[0:len(dataFile)-5]
            dataFile = [dataFile[i] for i in range(len(dataFile)) if CFFreq[i] <= 6000*1e6]
            Stop_freq = 6000*1e6
            CFFreq = [CFFreq[i] for i in range(len(CFFreq)) if CFFreq[i] <= 6000*1e6]
            
        elif Start_freq < 100*1e6:
            #dataFile = dataFile[1:len(dataFile)]
            Start_freq = 100*1e6
            dataFile = [dataFile[i] for i in range(len(dataFile)) if CFFreq[i] >= 100*1e6]
            CFFreq = [CFFreq[i] for i in range(len(CFFreq)) if CFFreq[i] >= 100*1e6]
            
        dataFile = [dataFile[i] for i in range(len(dataFile)) if CFFreq[i] <= 6000*1e6 and CFFreq[i] >= 100*1e6]
        te = [CFFreq[i] for i in range(len(CFFreq)) if CFFreq[i] <= 6000*1e6 and CFFreq[i] >= 100*1e6]             
        if zoom_Start_freq <= Start_freq:
            zoom_Start_freq_plot = zoom_Start_freq 
        elif zoom_Stop_freq >= Stop_freq:
            zoom_Stop_freq_plot = zoom_Stop_freq 
            
        for cnt, value in enumerate(te):
                 if value >= zoom_Start_freq_plot and value <= zoom_Stop_freq_plot:
                     #Start_freq = value -20*1e6
                     #Stop_freq = value -20*1e6
                     d.append(dataFile[cnt])
                     #h.append(headerFile[cnt])
                     counter.append(cnt)
                     
        headerInfo = [open(headerFile[val],'r').readlines() for count, val in enumerate(counter)]
        temp = [value[Cnt+1] for counter, value in enumerate(headerInfo) for Cnt, Val in enumerate(value) if Val == 'Center Frequency in Hz:\n']
        temp = [float(value[:-1]) for counter, value in enumerate(temp) if value.endswith('\n')]
        temp = np.array(temp).astype('float32')
        if len(temp) <= 2:
            
            original_data = self.loadDataFromFile(temp[0],d[0])
            self.calibrateData(original_data, temp[0]+20*1e6, temp[0]-20*1e6, color, scaling_factor)
    
            original_data = self.loadDataFromFile(temp[1],d[1])
            self.calibrateData(original_data, temp[1]+20*1e6, temp[1]-20*1e6, color, scaling_factor)
        else:   
            
            original_data = self.loadDataFromFile(temp[0],d[0])
            self.calibrateData(original_data, temp[0]+20*1e6, temp[0]-20*1e6, color, scaling_factor)
    
            original_data = self.loadDataFromFile(temp[1],d[1])
            self.calibrateData(original_data, temp[1]+20*1e6, temp[1]-20*1e6, color, scaling_factor)
            
            original_data = self.loadDataFromFile(temp[2],d[2])
            self.calibrateData(original_data, temp[2]+20*1e6, temp[2]-20*1e6, color, scaling_factor)
        
        self.fig_plot.set_ylim(self.zoom_min,self.zoom_max)
        self.fig_plot.set_xlim(zoom_Start_freq/1e6,zoom_Stop_freq/1e6)
        
        self.original = False
       # self.fig_plot.legend(['Original data'])
        self.canvas.draw()   
       
    def plot(self):
        self.figure = Figure(figsize=(10,10))
        self.fig_plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent.top_frame_plot)
        self.canvas.get_tk_widget().configure(background="#d469a3", highlightcolor='grey', highlightbackground='white')
        self.canvas.get_tk_widget().pack(side=BOTTOM, fill=X, padx=5, pady=5)
        self.toolbar = NavSelectToolbar(self.canvas,self.parent.top_frame_plot,self)
        self.toolbar.update()
        self.fig_plot.set_ylabel("Electrical Field Strength [dBuV/m]")#('Power [dBm]')
        self.fig_plot.set_xlabel("Frequency (MHz)" )#(resolution %.3f kHz)"%1)
    
    def clear_plot(self):
        self.canvas.get_tk_widget().destroy()
        self.toolbar.destroy()
        self.canvas = None
        self.plot()
        
    def clear_data(self):
        answer = tkMessageBox.askyesno(title="Clear Plot and Data", message="Are you sure you want to clear the loaded data?")
        if (answer):
            self.zoom_data = None
            self.zoom_data2 = None
            self.DATA = None
            self.DATA2 = None
            self.original = None
            self.zoom_trigger = None
            self.zoom_trigger1 = None
            self.zoom_trigger2 = None
            tkMessageBox.showwarning(title="Clear Plot and Data", message="Data has been deleted.")
            self.clear_plot()
            
    def clear_data_without(self):
        self.zoom_data = None
        self.zoom_data2 = None
        self.DATA = None
        self.DATA2 = None
        self.original = None
        self.zoom_trigger = None
        self.zoom_trigger1 = None
        self.zoom_trigger2 = None
        self.clear_plot()
            
    def new_res_BW(self):
        entries = {}
        for field in fields_zoom:
            row = Frame(self.parent.newRBW_button_frame)
            lab = Label(row, width=22, text=field, anchor='w')
            ent = Entry(row, width=22,bg="lightblue",
                                    fg="black",highlightcolor="lightblue",insertbackground="black",
                                    highlightthickness=2)
            row.pack(side=TOP, fill=X, padx=2, pady=2)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT, expand=NO, fill=X)
            entries[field] = ent
        return entries
    
        
    def launch_dir_finder(self):
        #directory = tkFileDialog.askdirectory()
        files = tkFileDialog.askopenfilenames()
        self.directory_entry.delete(0,END)
        self.directory_entry.insert(0,files)
        
    def _custom_button(self,root,text,command,**kwargs):
        button = Button(root, text=text,
            command=command,padx=2, pady=2,height=1, width=10,**kwargs)
        button.pack(side=TOP,fill=BOTH)
        return button
    
    def quit(self):
        msg = "Quitting:\nUnsaved progress will be lost.\nDo you wish to Continue?"
        if tkMessageBox.askokcancel("Combustible Lemon",msg):
            self.parent.root.destroy()    

   
    def cal_GainCircuit(self, upperfreq, lowerfreq, scaling_factor = 40): 
       self.GainCircuitData = self.GainLNA_DatafileCSV(self.LNApath)
       freqGain = (self.GainCircuitData[0,:])
       newfreqGain = np.linspace(lowerfreq,upperfreq,scaling_factor)
       testGain = np.interp(newfreqGain, freqGain, self.GainCircuitData[1,:])
       Gaintemp.append([newfreqGain, testGain])
       return testGain 

     
    def calCCFdBuvPerM(self,spectrum, CCF, Lcable, G_LNA, antennaEfficiency): # returns in [dBuV/m]
         # spectrum numpy array
         #get the 1MHz average of the spectrum
         Z0 = 377  # Impedance of freespace
         r = 1.0 # Distance DUT to Antenna
         antennaEfficiency = 0.75
         
         temp = -CCF-G_LNA + Lcable - (10.0 * np.log10(antennaEfficiency)) + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r*r)))) + 90.0
         
         return spectrum[0], spectrum[1]+temp, spectrum[2], spectrum[3]+temp#, spectrum[3]+temp   
     
    def calCCFdBuvPerM_Original(self,spectrum, CCF, Lcable, G_LNA, antennaEfficiency): # returns in [dBuV/m]
         # spectrum numpy array
         #get the 1MHz average of the spectrum
         Z0 = 377  # Impedance of freespace
         r = 1.0 # Distance DUT to Antenna
         antennaEfficiency = 0.75
         
         temp = -CCF-G_LNA + Lcable - (10.0 * np.log10(antennaEfficiency)) + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r*r)))) + 90.0
     #    test.append([spectrum[0], spectrum[1]+temp, G_LNA, CCF])
         return spectrum[0], spectrum[1]+temp 
     
    def calibrateData(self, reduced_data, Stop_freq, Start_freq, color_data_set, scaling_factor):
       
       CCF = self.get_CCF(scaling_factor, Stop_freq, Start_freq)
       G_LNA = self.cal_GainCircuit(Stop_freq, Start_freq, scaling_factor)

       
       ylabel = "Electrical Field Strength [dBuV/m]"
       if self.original == True:
           Spec = self.calCCFdBuvPerM_Original(reduced_data, CCF, self.Lcable, G_LNA, self.antennaEfficiency)
           self.plot_data(Spec,ylabel,color_data_set, Start_freq, Stop_freq)
           #self.original = False
       else:
           Spec = self.calCCFdBuvPerM(reduced_data, CCF, self.Lcable, G_LNA, self.antennaEfficiency)
           self.plot_data(Spec,ylabel,color_data_set, Start_freq, Stop_freq)
       
       return Spec
       
    def read_reduce_Data(self,spectrum,scaling_factor):
        # read in the whole BW in one array
# set the display sample size depending on the display bandwidth and resolution  
        
        spec = self.dBuV_M2V_M(spectrum[1])
        freq = spectrum[0]
        x = int(len(spec)/scaling_factor)
        spec_min = np.array([], dtype=np.float32)
        spec_max = np.array([], dtype=np.float32)
        freq_min = np.array([], dtype=np.float32)
        freq_max = np.array([], dtype=np.float32)
        
        spec_max = [np.max(spec[(i*x):(x*i+x)]) for i in range (scaling_factor)]
        ind_max  = [(np.argmax(spec[(i*x):(x*i+x)])+x*i) for i in range (scaling_factor)]
        spec_min = [np.min(spec[(i*x):(x*i+x)]) for i in range (scaling_factor)]
        ind_min  = [(np.argmin(spec[(i*x):(x*i+x)])+x*i) for i in range (scaling_factor)]
        freq_max = [freq[value] for count, value in enumerate(ind_max)]
        freq_min = [freq[value] for count, value in enumerate(ind_min)]
            #self.leg_data = [spec_max, spec_min] 
        spec_max = self.V_M2dBuV_M(spec_max)
        spec_min = self.V_M2dBuV_M(spec_min)
       # freq = spectrum[0]#np.linspace(Start_freq,Stop_freq,len(spec_max)) 
       
        temp = freq_max,spec_max,freq_min,spec_min
        data = np.array(temp, dtype=np.float32)
        return data       
    
    def dBuV_M2V_M(self,spec):
        VperM = pow(10,(spec-120)/20)
        return VperM    
    
    def V_M2dBuV_M(self,spec):
        dBuV_M = 20*np.log10(spec)+120
        return dBuV_M  
    
    def getZoomInput(self):
        if self.newRWB_entry_data["Start Frequency (MHz): "].get() == '':
            tkMessageBox.showwarning(title="Warning", message="Please enter the Start Frequency in MHz")

        elif self.newRWB_entry_data["Stop Frequency (MHz): "].get() == '':
            tkMessageBox.showwarning(title="Warning", message="Please enter the Stop Frequency in MHz")
        elif self.newRWB_entry_data["Maximum amplitude: "].get() == '':
            tkMessageBox.showwarning(title="Warning", message="Please enter the Maximum amplitude")
        elif self.newRWB_entry_data["Minimum amplitude: "].get() == '':
            tkMessageBox.showwarning(title="Warning", message="Please enter the Minimum amplitude")
        else:
            self.zoom_Start_freq = float(self.newRWB_entry_data["Start Frequency (MHz): "].get())*1e6
            self.zoom_Stop_freq = float(self.newRWB_entry_data["Stop Frequency (MHz): "].get())*1e6
            self.zoom_max = float(self.newRWB_entry_data["Maximum amplitude: "].get())
            self.zoom_min = float(self.newRWB_entry_data["Minimum amplitude: "].get())

    def zoomActivate(self):
        if self.zoom_trigger1 == True:
            self.clear_plot()

        if self.zoom_trigger2:
            #self.clear_plot()
            self.zoom_trigger2 = False
        else:
            self.clear_data_without()
            
            
        self.zoom_trigger = True
        self.zoom_trigger1 = True
        self.getZoomInput()
        zoom_Bandwidth = self.zoom_Stop_freq - self.zoom_Start_freq
        zoom_nr_smp = int(zoom_Bandwidth/(40*1e6))
        if zoom_Bandwidth >= 35*1e6:
            if zoom_nr_smp > 111:
                scaling_factor = 40
                self.zoom_dump_data(self.zoom_Start_freq, self.zoom_Stop_freq,scaling_factor, color_data_set1, self.dataFile, self.CFreq, self.Start_freq, self.Stop_freq)
            elif zoom_nr_smp < 111 and zoom_nr_smp >= 74:
                scaling_factor = 800
                self.zoom_dump_data(self.zoom_Start_freq, self.zoom_Stop_freq,scaling_factor, color_data_set1, self.dataFile, self.CFreq, self.Start_freq, self.Stop_freq)
            elif zoom_nr_smp < 74 and zoom_nr_smp >= 37:
                scaling_factor = 10000
                self.zoom_dump_data(self.zoom_Start_freq, self.zoom_Stop_freq,scaling_factor, color_data_set1, self.dataFile, self.CFreq, self.Start_freq, self.Stop_freq)
            elif zoom_nr_smp < 37 and zoom_nr_smp >= 1 and zoom_Bandwidth > 40*1e6:
                scaling_factor = 15000
                self.zoom_dump_data(self.zoom_Start_freq, self.zoom_Stop_freq,scaling_factor, color_data_set1, self.dataFile, self.CFreq, self.Start_freq, self.Stop_freq)
            elif zoom_Bandwidth <= 40*1e6:#zoom_nr_smp <= 1:
                #plot orginal    
                self.original = True
                scaling_factor = 200000
                # create a seperate original function
                self.zoom_dump_data_org(self.zoom_Start_freq, self.zoom_Stop_freq, color_data_set1[3],self.headerFile, self.dataFile, self.CFreq, self.Start_freq, self.Stop_freq)
                
        else: 
            tkMessageBox.showwarning(title="Warning", message="Maximum zoom bandwidth must be more than acquisition bandwidth which is %d MHz."%(int(round(self.Bandwidth/1e6))))

        
    def zoomActivate2Data(self):
        if self.zoom_trigger2 == True:
            self.clear_plot()

        if self.zoom_trigger1:
            #self.clear_plot()
            self.zoom_trigger1 = False
        else:
            self.clear_data_without()
            
            
        self.zoom_trigger = True
        self.zoom_trigger2 = True
        self.getZoomInput()
        zoom_Bandwidth = self.zoom_Stop_freq - self.zoom_Start_freq
        zoom_nr_smp = int(zoom_Bandwidth/(40*1e6))
        if zoom_Bandwidth >= 35*1e6:
            if zoom_nr_smp > 111:
                scaling_factor = 40
                self.zoom_dump_data(self.zoom_Start_freq, self.zoom_Stop_freq,scaling_factor, color_data_set2, self.dataFile2, self.CFreq2, self.Start_freq2, self.Stop_freq2)
            elif zoom_nr_smp < 111 and zoom_nr_smp >= 74:
                scaling_factor = 800
                self.zoom_dump_data(self.zoom_Start_freq, self.zoom_Stop_freq,scaling_factor, color_data_set2, self.dataFile2, self.CFreq2, self.Start_freq2, self.Stop_freq2)
            elif zoom_nr_smp < 74 and zoom_nr_smp >= 37:
                scaling_factor = 10000
                self.zoom_dump_data(self.zoom_Start_freq, self.zoom_Stop_freq,scaling_factor, color_data_set2, self.dataFile2, self.CFreq2, self.Start_freq2, self.Stop_freq2)
            elif zoom_nr_smp < 37 and zoom_nr_smp > 1:
                scaling_factor = 15000
                self.zoom_dump_data(self.zoom_Start_freq, self.zoom_Stop_freq,scaling_factor, color_data_set2, self.dataFile2, self.CFreq2, self.Start_freq2, self.Stop_freq2)
            elif zoom_Bandwidth <= 40*1e6:
                #plot orginal      
                self.original = True
                # create a seperate original function
                self.zoom_dump_data_org(self.zoom_Start_freq, self.zoom_Stop_freq, color_data_set2[3],self.headerFile2, self.dataFile2, self.CFreq2, self.Start_freq2, self.Stop_freq2)
                
        else: 
            tkMessageBox.showwarning(title="Warning", message="Maximum zoom bandwidth must be more than acquisition bandwidth which is %d MHz."%(int(round(self.Bandwidth/1e6))))

      
    def plot_data(self,reduced_data,yaxis_label,color, Start_freq, Stop_freq):
        if self.original == True:
            self.org_data = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[1], color=color)
            self.leg = ['Original']
        else:
            if self.zoom_trigger1 == True:
                self.zoom_trigger = False 
                self.max_plot, = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[1], color=color[0])
                self.min_plot, = self.fig_plot.plot(reduced_data[2]/1e6,reduced_data[3], color=color[1])
                self.leg = ['Max', 'Min']
                self.leg_data = [self.max_plot, self.min_plot]
            elif self.zoom_trigger2 == True:
                self.zoom_trigger = False 
                self.max_plot, = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[1], color=color[0])
                self.min_plot, = self.fig_plot.plot(reduced_data[2]/1e6,reduced_data[3], color=color[1])
                self.leg = ['Max', 'Min']              
                self.leg_data = [self.max_plot, self.min_plot]
                
            else:
                self.max_plot, = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[1], color=color[0])
                self.min_plot, = self.fig_plot.plot(reduced_data[2]/1e6,reduced_data[3], color=color[1])
                self.leg = ['Max', 'Min']
                self.leg_data = [self.max_plot, self.min_plot]

            
class CandidateFinder(object):
    def __init__(self):
        self.filenames = []
        self.counter = None

    def _is_valid(self,pfd):
        ps_valid = os.path.isfile("%s.ps" % pfd)
        bp_valid = os.path.isfile("%s.bestprof" % pfd)
        return all((ps_valid,bp_valid))

    def get_from_directory(self,directory):
        pfds = glob.glob("%s/*.pfd" % directory)
        print ("%s/*.pfd" % directory)
        if not pfds:
            return None
        for pfd in pfds:
            if self._is_valid(pfd):
                self.filenames.append(pfd)

    def get_from_directories(self,directory):
        counter = 0
        print ("Searching %s" % directory)
        rambler = os.walk(directory)
        for path,dirnames,filenames in rambler:
            for filename in filenames:
                if filename.endswith(".pfd"):
                    pfd = os.path.join(path,filename)
                    if self._is_valid(pfd):
                        self.filenames.append(pfd)
                        counter+=1
                        sys.stdout.write("Found %d files...\r"%counter)
                        sys.stdout.flush()
                
    def parse_all(self):
        filenames = list(set(self.filenames))
        nfiles = len(filenames)
        recarray = np.recarray(nfiles,dtype=BESTPROF_DTYPE)
        print ("Parsing %d .bestprof files..." % nfiles)
        for ii,filename in enumerate(filenames):
            if ii%10 == 0:
                sys.stdout.write("%.2f\r"%(100.*ii/nfiles))
                sys.stdout.flush()
            bestprof_file = "%s.bestprof" % filename
            info = parse_bestprof(bestprof_file)
            for key in PLOTABLE_FIELDS:
                if key in info.keys():
                    val = info[key]
                else: 
                    val = 0
                recarray[ii][key] = val
            recarray[ii]["PFD_file"] = filename
        return recarray
    
    def parse_bestprof(filename):
        f = open(filename,"r")
        lines = f.readlines()
        f.close()
        info = {} 
        for ii,line in enumerate(lines):
            if not line.startswith("# "):
                continue
            if line.startswith("# Prob(Noise)"):
                line = line[2:].split("<")
            else:
                line = line[2:].split("=")
                
            key = line[0].strip()
            value = line[1].strip()
            
            if "+/-" in value:
                value = value.split("+/-")[0]
                if "inf" in value:
                    value = "0.0"
    
            if value == "N/A":
                value = "0.0"
    
            if "Epoch" in key:
                key = key.split()[0]
    
            if key == "Prob(Noise)":
                key = "Sigma"
                try:
                    value = value.split("(")[1].split()[0].strip("~")
                except:
                    value = "30.0"
                        
            info[key]=value
        return info
 
    
if __name__ == '__main__':
   root = Tk()
   root.tk_setPalette(**DEFAULT_PALETTE)
   start = GUI_set_up(root)

   root.mainloop() 