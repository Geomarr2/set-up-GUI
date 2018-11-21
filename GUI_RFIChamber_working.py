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
matplotlib.use('cairo')
from IPython import get_ipython
#get_ipython().magic('reset -sf')
import numpy as np
from io import StringIO
global fields 
import calData as cal
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
#import matplotlib as plt

from matplotlib.widgets import Lasso
from matplotlib.figure import Figure
import RFIHeaderFile
#from tkinter import Label, Button, Radiobutton, IntVar

FileCCF="CCF4.csv"
PathCCF = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/Spectrum/CCF4.csv"
PathGainCircuit = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/RFIChamberMeasure/RFcable/GainCircuit.csv"
FileGainCircuit = "GainCircuit.csv"
fields = 'Path', 'Filename', 'Start Frequency (MHz)', 'Stop Frequency (MHz)', 'LNA gain (dB)', 'Cable losses (dB)','Antenna efficiency'
fields_zoom = "Start Frequency (MHz): ", "Stop Frequency (MHz): ","Maximum amplitude: ","Minimum amplitude: "
AcqBW = 40e6
color_data_set1 = ['r','b','m']
color_data_set2 = ['hotpink','c','g']
global org_data 
bandwidth = AcqBW #Hz
#StartFreq = 1000e6
#StopFreq = 1200e6

#Lcable = -1  #dB cable losses
#antennaEfficiency = 0.75 
Nsample = 523852
r = 1

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
    "figure.facecolor":"black",
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
        self.Filename = "Spectrum_"
        self.CenterFrequency = 0.
        self.Bandwidth = 40*1e6
        self.Nchannels = 373851
        self.G_LNA = 0. 
        self.Lcable = 0.
        self.antennaEfficiency = 0.
        self.data = None
        self.scaling_factor=0.           # number of points full data set
        
        # set counters
        self.missingData = 0
        self.UserInputMissingData = 0
        
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
        
        self.zoom_data = None
        self.original_data = None
        self.zoom_data2 = None
        self.original_data2 = None
        
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
        
        
        self.LNApath = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/RFIChamberMeasure/RFcable/GainCircuitNew.csv"
        self.CCFpath = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/Spectrum/CCF4.csv"
        self.GainCircuitData = self.GainLNA_DatafileCSV(self.LNApath)
        
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
        
        
        
        
    def printHeaderInfo(self):
#Load in all the header file get the info and the BW from max and min center freq
        msg = []
        file = [value for counter, value in enumerate(self.headerFile)]
        self.headerInfo = [open(self.headerFile[count],'r').readlines() for count, val in enumerate(self.headerFile)]
        temp = [value[Cnt+1] for counter, value in enumerate(self.headerInfo) for Cnt, Val in enumerate(value) if Val == 'Unique Scan ID:\n']
        temp = [value[:-1] for counter, value in enumerate(temp) if value.endswith('\n')]
             
#check if all the numpy arrays is there
        spec = np.array([], dtype=np.float32)
        temp = [np.load(value) if value.find(valueHeader) > 0 else self.nextCommand() for counter, value in enumerate(self.dataFile) for counterHeader, valueHeader in enumerate(temp)]
        return temp
    
    def askUser(self):
        self.printHeaderInfo()
        if self.missingData == 1:
            if tkMessageBox.askquestion('Warning','Data is missing: would you like to reload a new data set?') == 'yes':
                self.printHeaderInfo()
            else:
                result = self.askUserMultipleQuestion("What would you like to plot?",
                                                      ["1. Continue the current plot without the data set",
                                                              "2. Plot only until missing data set",
                                                              "3. Would you like to replot new data set"])
                
        
            self.missingData = 0
        else:
            tkMessageBox.showinfo(title = "Acquisition information", message = msg)
        
        

    def askUserMultipleQuestion2(self,prompt, options):
        root2 = Toplevel(self.root,bg="gray90")
        #root2.title("Continue with missing data set")
        label_frame = Frame(root2)
        label_frame.pack(side=TOP,fill=BOTH,expand=1)
        button_frame = Frame(root2)
        button_frame.pack(side=BOTTOM,fill=BOTH,expand=1)

        if prompt:
            Label(label_frame, text=prompt).pack()
        v = IntVar()
        for i, option in enumerate(options):
            Radiobutoon(button_frame, text=option, variable=v, value=i, selectcolor="gray90").pack(anchor="w")
        button_submit_frame = Frame(button_frame)
        button_submit_frame.pack(side=BOTTOM,fill=BOTH,expand=1)
        Button(button_submit_frame,text="Submit", command=root2.destroy,**DEFAULT_STYLE_NEW).pack()
        
        if v.get() == 0: 
            returnVal = None
        else:
            returnVal = options[v.get()]
        print("User's response was: {}".format(repr(returnVal)))
        return returnVal 
    
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
      
        
        
    def printHeaderInfo(self,dataFile,headerFile):
#Load in all the header file get the info and the BW from max and min center freq

        file = [value for counter, value in enumerate(headerFile)]
        headerInfo = [open(headerFile[count],'r').readlines() for count, val in enumerate(headerFile)]
        temp = [value[Cnt+1] for counter, value in enumerate(headerInfo) for Cnt, Val in enumerate(value) if Val == 'Center Frequency in Hz:\n']
        temp = [float(value[:-1]) for counter, value in enumerate(temp) if value.endswith('\n')]
        temp = np.array(temp).astype('float32')
#check if all the numpy arrays is there
        if len(temp) == len(dataFile):
            #temp = [np.load(value) for counter, value in enumerate(self.dataFile) for counterHeader, valueHeader in enumerate(temp) if value == (valueHeader+'.npy')]
            file2 = open(headerFile[0], "r")
            msg = []
            
            headerInfo = file2.readlines()
            if headerInfo[6] != 'default\n' and headerInfo[18] != 'default\n' and headerInfo[20] != 'default\n'and headerInfo[22] != 'default\n'and headerInfo[24] != 'default\n':
                self.Bandwidth = float(headerInfo[6])
                self.CCFpath = headerInfo[18]
                self.antennaEfficiency = float(headerInfo[20])
                self.Lcable = float(headerInfo[22])
                self.LNApath = headerInfo[24]
            else:
                self.Bandwidth = 40*1e6
                self.LNApath = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/RFIChamberMeasure/RFcable/GainCircuitNew.csv"
                self.CCFpath = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/Spectrum/CCF4.csv"
                self.antennaEfficiency = 0.75
                self.Lcable = -1
            for x in headerInfo:
                msg.append(x)
            file2.close()
            tkMessageBox.showinfo(title = "Acquisition information", message = msg)
        else:
            tkMessageBox.showwarning('Warning','Data is missing')
           
        return msg, temp

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
        with open(path, "r") as filedata:
            csvdata = np.genfromtxt(filedata, delimiter = ',')
            for row in csvdata:
                data.append(row)
            filedata.close()
            data =np.array(data, dtype = 'float')    
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
        
    def findCenterFrequency(self,Bandwidth):
#Load in all the header file get the info and the BW from max and min center freq
        center_freq_temp = []
        for i in range(len(self.headerFile)): # i = 40 Mhz range
            temp_path = self.headerFile[i].split('_')
            name_file = temp_path[0]
            cfreq = temp_path[1]
            scanID_file = temp_path[2]
            center_freq_temp.append(float(cfreq))
            
        Stop_freq = max(center_freq_temp) + Bandwidth/2
        Start_freq = min(center_freq_temp) - Bandwidth/2   
        #print('Display %f - %f Hz\n' %(Start_freq, Stop_freq))
        self.CenterFrequency = Start_freq+(Stop_freq-Start_freq)/2
       # print('With a center frequency of %f'%self.CenterFrequency)
        return Start_freq,Stop_freq
    
    def dump_filenames(self):
#Load in vereything and seperate header and data file
        if self.zoom_trigger1 == True or self.zoom_trigger2 == True:
            self.clear_plot()
            self.zoom_trigger1 = False
            self.zoom_trigger2 = False
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
        msg,temp = self.printHeaderInfo(dataFile,headerFile)
        self.Bandwidth = float(msg[6])
        Start_freq = min(temp) - self.Bandwidth/2
        Stop_freq = max(temp) + self.Bandwidth/2
        self.CenterFrequency = (Stop_freq-Start_freq)/2
        print('Start Frequency: %f\n' %Start_freq)
        print('Stop Frequency: %f\n' %Stop_freq)
        #self.findCenterFrequency(self.Bandwidth)
        self.Start_freq = Start_freq
        self.Stop_freq = Stop_freq
        self.original_data = self.loadDataFromFile(Start_freq,Stop_freq,dataFile)
        self.scaling_factor = 500
        self.calibrateData(self.read_reduce_Data(self.original_data[1], Start_freq, Stop_freq,self.scaling_factor), Stop_freq, Start_freq, color_data_set1)

    def dump_filenames_data2(self):
#Load in vereything and seperate header and data file
        if self.zoom_trigger1 == True or self.zoom_trigger2 == True:
            self.clear_plot()
            self.zoom_trigger1 = False
            self.zoom_trigger2 = False
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
        self.zoom_trigger1 = False
        self.zoom_trigger2 = False
        msg,temp = self.printHeaderInfo(dataFile,headerFile)
        self.Bandwidth2 = float(msg[6])
        Start_freq = min(temp) - self.Bandwidth/2
        Stop_freq = max(temp) + self.Bandwidth/2
        self.CenterFrequency = (Stop_freq-Start_freq)/2
        #self.findCenterFrequency(self.Bandwidth2)
        print('Start Frequency: %f\n' %Start_freq)
        print('Stop Frequency: %f\n' %Stop_freq)
        self.Start_freq2 = Start_freq
        self.Stop_freq2 = Stop_freq
        self.original_data2 = self.loadDataFromFile(Start_freq,Stop_freq,dataFile)
        self.scaling_factor2 = 500
        self.calibrateData(self.read_reduce_Data(self.original_data2[1], Start_freq, Stop_freq,self.scaling_factor2), Stop_freq, Start_freq, color_data_set2)

             
    def plot(self):
        self.figure = Figure(figsize=(10,10))
        self.fig_plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent.top_frame_plot)
        self.canvas.get_tk_widget().pack(side=BOTTOM, fill=X, padx=5, pady=5)
        self.fig_plot = self.figure.add_subplot(111)
        self.toolbar = NavSelectToolbar(self.canvas,self.parent.top_frame_plot,self)
        self.toolbar.update()
    
    
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
            tkMessageBox.showwarning(title="Clear Plot and Data", message="Data has been deleted.")
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

    def GainLNA_DatafileCSV(self,path):
        data = np.array([])  
        temp_data = np.array([]) 
        if path.endswith('\n'):
            path = path[:-1]
        f = open(path, "r")
        row = f.readlines()
        splitRow = [row[i].split(';') for i in range(len(row))]
            #csvdata = np.load(filedata, delimiter = ',', skip_header=1)
        data = [value for counter, value in enumerate(splitRow)]#(np.append(data, row) for row in csvdata)
        
        newData = [data[i] for i in range(len(data)) if 2 <= len(data[i])]
        newDataTemp = np.array([newData[i] for i in range(1,len(newData))])
        freq = newDataTemp[:,0]
        dataG = (newDataTemp[:,1])
        freq = [float(i.replace(',','.')) for i in freq]
        dataG = [float(i.replace(',','.')) for i in dataG]
        GainLNA = np.array([freq,dataG]).astype('float32')
        return GainLNA

    def makeform(self):
       entries = {}
       for field in fields:
           row = Frame(elf.parent.input_frame)
           lab = Label(row, width=22, text=field+": ", anchor='w')
           ent = Entry(row, width=22)
           # ent.insert(0,"0")
           row.pack(side=TOP, fill=X, padx=5, pady=5)
           lab.pack(side=LEFT)
           ent.pack(side=RIGHT, expand=NO, fill=X)
           entries[field] = ent 
           print(field+': %s \n' % (entries[field].get()))
       return entries
   
    def get_CCF(self, upperfreq, lowerfreq): 
       CCFdata = self.CCF_DatafileCSV(self.CCFpath)
       bw = (upperfreq-lowerfreq)/1e6
       x = int(len(CCFdata)/bw)
       temp_spec = np.array([], dtype=np.float32)
       temp = 0
       CCFFreq = CCFdata[:,0]
       CCFTemp = CCFdata[:,1]
       temp = CCFTemp[(CCFFreq>=lowerfreq)&(CCFFreq<=upperfreq)]
       meantemp = np.mean(temp)
       return meantemp.astype('float32')  
   
    def reduceMean(self,spectrum):
       spec_mean = np.array([], dtype=np.float32)
       x = len(spectrum)/self.scaling_factor
       for i in range(self.scaling_factor): 
           temp_spec_mean = np.mean(spectrum[i*x:x*(i+1)])
           spec_mean = np.append(spec_mean, temp_spec_mean)
       return spec_mean.astype('float32') 
   
    def cal_GainCircuit(self, upperfreq, lowerfreq, array_size = 40): 
       #upperfreq = self.Stop_freq
       #lowerfreq =self.Start_freq
       temp_gain = np.array([], dtype=np.float32)
       temp = 0
       freqGain = (self.GainCircuitData[0,:])
       if len(freqGain) > 1:
            count = 0
            for j in range(len(freqGain)):
               if lowerfreq <= freqGain[j] and upperfreq >= freqGain[j]:
                   temp = self.GainCircuitData[1,j] + temp
                   count = count + 1
            temp = temp/count
            temp_gain = np.append(temp_gain, temp)
       else:
           for j in range(len(freqGain)):
               if lowerfreq <= freqGain[j] and upperfreq >= freqGain[j]:
                   temp = self.GainCircuitData[1,j] 
                   temp_gain = np.append(temp_gain, temp)
       return (temp_gain.T).astype('float32') 

    def calCCFpower(self,spectrum, CCF, r, Lcable, G_LNA, antennaEfficiency): # returns in [dBm]
         # spectrum numpy array
         temp = -G_LNA - Lcable - (10.0 * np.log10(antennaEfficiency)) - CCF 
         return spectrum[0], spectrum[1]+temp
     
    def calCCFdBuvPerM(self,spectrum, CCF, Lcable, G_LNA, antennaEfficiency): # returns in [dBuV/m]
         # spectrum numpy array
         #get the 1MHz average of the spectrum
         Z0 = 119.9169832 * np.pi  # Impedance of freespace
         r = 1.0 # Distance DUT to Antenna
         temp = -G_LNA - Lcable - (10.0 * np.log10(antennaEfficiency)) - CCF + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r*r)))) + 90.0

         return spectrum[0], spectrum[1]+temp, spectrum[2]+temp, spectrum[3]+temp   
     
    def calCCFdBuvPerMOrg(self,spectrum, CCF, Lcable, G_LNA, antennaEfficiency): # returns in [dBuV/m]
         # spectrum numpy array
         #get the 1MHz average of the spectrum
         Z0 = 119.9169832 * np.pi  # Impedance of freespace
         r = 1.0 # Distance DUT to Antenna
         temp = -G_LNA - Lcable - (10.0 * np.log10(antennaEfficiency)) - CCF + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r*r)))) + 90.0

         return spectrum[0], spectrum[1]+temp 
     
    def calibrateData(self, reduced_data, upperfreq, lowerfreq,color_data_set):
       tstart = time.time()
       #self.GainCircuitData = self.readIQDatafileCSV(self.LNApath)
       G_LNA = self.cal_GainCircuit(upperfreq, lowerfreq)
       CCF = self.get_CCF(upperfreq, lowerfreq)
       #CCF = self.reduceMean(CCF)
       if self.original == True:
           Spec = self.calCCFdBuvPerMOrg(reduced_data, CCF, self.Lcable, G_LNA, self.antennaEfficiency)
       else:
           Spec = self.calCCFdBuvPerM(reduced_data, CCF, self.Lcable, G_LNA, self.antennaEfficiency)
       ylabel = 'Power [dBm]'
       ylabel = "Electrical Field Strength [dBuV/m]"
       self.plot_data(Spec,ylabel,color_data_set, lowerfreq, upperfreq)
       print ('time:' , (time.time()-tstart))
       
    def read_reduce_Data(self,spectrum,Start_freq, Stop_freq,scaling_factor):
        # read in the whole BW in one array
# set the display sample size depending on the display bandwidth and resolution  
        spectrum = self.dBuV_M2V_M(spectrum)
        x = int(len(spectrum)/scaling_factor)
        spec_mean = np.array([], dtype=np.float32)
        spec_min = np.array([], dtype=np.float32)
        spec_max = np.array([], dtype=np.float32)
        freq = np.array([], dtype=np.float32)
        spec_max = [np.max(spectrum[i*x:x*(i+1)]) for i in range (scaling_factor)]
        spec_mean = [np.mean(spectrum[i*x:x*(i+1)]) for i in range (scaling_factor)]
        spec_min = [np.min(spectrum[i*x:x*(i+1)]) for i in range (scaling_factor)]
        spec_max = self.V_M2dBuV_M(spec_max)
        spec_mean = self.V_M2dBuV_M(spec_mean)
        spec_min = self.V_M2dBuV_M(spec_min)
        freq = np.linspace(Start_freq,Stop_freq,len(spec_max))        
        temp = freq,spec_max,spec_min,spec_mean
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
        #if self.zoom_trigger1 == True:
         #   self.clear_plot()
        self.zoom_trigger = True
        self.zoom_trigger1 = True
        self.getZoomInput()
        #if self.zoom_trigger2:
         #   self.zoom_trigger2 = False
        #else:
         #   self.clear_plot()
        # save new zoom data
        org_freq = self.original_data[0,:]
        org_data = self.original_data[1,:]
        self.zoom_data = org_data[(org_freq>=self.zoom_Start_freq)&(org_freq<=self.zoom_Stop_freq)]
        self.scaling_factor = int((self.Stop_freq-self.Start_freq)/1e6)
        max_nr_smp = int((self.Stop_freq - self.Start_freq)/self.Bandwidth)
        print(max_nr_smp)
        zoom_Bandwidth = self.zoom_Stop_freq - self.zoom_Start_freq
        print(zoom_Bandwidth)
        zoom_nr_smp = int(zoom_Bandwidth/(40*1e6))
        print(zoom_nr_smp)
        if zoom_Bandwidth >= 35*1e6:
            if zoom_nr_smp > 111:
                self.scaling_factor = int(zoom_Bandwidth/1e6)
                reduced_data = self.read_reduce_Data(self.zoom_data, self.zoom_Start_freq, self.zoom_Stop_freq,self.scaling_factor)
            elif zoom_nr_smp < 111 and zoom_nr_smp >= 74:
                self.scaling_factor = int(30*zoom_Bandwidth/1e6)
                reduced_data = self.read_reduce_Data(self.zoom_data, self.zoom_Start_freq, self.zoom_Stop_freq,self.scaling_factor)
            elif zoom_nr_smp < 74 and zoom_nr_smp >= 37:
                self.scaling_factor = int((50)*zoom_Bandwidth/1e6)
                reduced_data = self.read_reduce_Data(self.zoom_data, self.zoom_Start_freq, self.zoom_Stop_freq,self.scaling_factor)
            elif zoom_nr_smp < 37 and zoom_nr_smp >= 2:
                self.scaling_factor = int((100)*zoom_Bandwidth/1e6)
                reduced_data = self.read_reduce_Data(self.zoom_data, self.zoom_Start_freq, self.zoom_Stop_freq,self.scaling_factor)
            elif zoom_nr_smp <= 1:
                #plot orginal
                #self.scaling_factor = int(zoom_Bandwidth/1e6)*1000
                spec = self.zoom_data
                freq = np.linspace(self.zoom_Start_freq,self.zoom_Stop_freq,len(spec))        
                temp = freq,spec
                self.original = True
                reduced_data = np.array(temp, dtype=np.float32)
            
            self.calibrateData(reduced_data, self.zoom_Stop_freq, self.zoom_Start_freq,color_data_set1)
        else: 
            tkMessageBox.showwarning(title="Warning", message="Maximum zoom bandwidth must be more than acquisition bandwidth which is %d MHz."%(int(round(self.Bandwidth/1e6))))
    
    def zoomActivate2Data(self):
       # if self.zoom_trigger2 == True:
        #    self.clear_plot()
        self.getZoomInput()
        self.zoom_trigger = True
        self.zoom_trigger2 = True
       # if self.zoom_trigger1:
        #    self.zoom_trigger1 = False
       # else:
        #    self.clear_plot()
        # save new zoom data
        org_freq = self.original_data2[0,:]
        org_data = self.original_data2[1,:]
        self.zoom_data2 = org_data[(org_freq>=self.zoom_Start_freq)&(org_freq<=self.zoom_Stop_freq)]
        self.scaling_factor2 = int((self.Stop_freq2-self.Start_freq2)/1e6)
        max_nr_smp = int((self.Stop_freq2 - self.Start_freq2)/self.Bandwidth2)
        zoom_Bandwidth = self.zoom_Stop_freq - self.zoom_Start_freq
        if zoom_Bandwidth >= 35*1e6:
            zoom_nr_smp = int(zoom_Bandwidth/(40*1e6))
            if zoom_nr_smp > 111:
                self.scaling_factor2 = int(zoom_Bandwidth/1e6)
                reduced_data = self.read_reduce_Data(self.zoom_data2, self.zoom_Start_freq, self.zoom_Stop_freq,self.scaling_factor2)
            elif zoom_nr_smp < 111 and zoom_nr_smp >= 74:
                self.scaling_factor = int(30*zoom_Bandwidth/1e6)
                reduced_data = self.read_reduce_Data(self.zoom_data2, self.zoom_Start_freq, self.zoom_Stop_freq,self.scaling_factor2)
            elif zoom_nr_smp < 74 and zoom_nr_smp >= 37:
                self.scaling_factor2 = int((50)*zoom_Bandwidth/1e6)
                reduced_data = self.read_reduce_Data(self.zoom_data2, self.zoom_Start_freq, self.zoom_Stop_freq,self.scaling_factor2)
            elif zoom_nr_smp < 37 and zoom_nr_smp >= 2:
                self.scaling_factor2 = int((100)*zoom_Bandwidth/1e6)
                reduced_data = self.read_reduce_Data(self.zoom_data2, self.zoom_Start_freq, self.zoom_Stop_freq,self.scaling_factor2)
            elif zoom_nr_smp <= 1:
                spec = self.zoom_data2
                freq = np.linspace(self.zoom_Start_freq,self.zoom_Stop_freq,len(spec))        
                temp = freq,spec
                self.original = True
                reduced_data = np.array(temp, dtype=np.float32)
                
            self.calibrateData(reduced_data, self.zoom_Stop_freq, self.zoom_Start_freq,color_data_set2)
        else: 
            tkMessageBox.showwarning(title="Warning", message="Maximum zoom bandwidth must be more than acquisition bandwidth which is %d MHz."%(int(round(self.Bandwidth/1e6))))
         
    def plot_data(self,reduced_data,yaxis_label,color_data_set, Start_freq, Stop_freq):
        #reduced_data = self.read_reduce_Data(data) 
        if self.original == True:
            self.original = False
            data = reduced_data
            self.fig_plot.plot(data[0]/1e6,data[1], color=color_data_set[0])
            self.fig_plot.set_ylim(self.zoom_min,self.zoom_max)
            self.fig_plot.set_xlim(self.zoom_Start_freq/1e6,self.zoom_Stop_freq/1e6)
            self.fig_plot.legend(['Original data'])
            self.canvas.draw()
        else:
            if self.zoom_trigger1 == True:
                self.fig_plot.set_ylim(self.zoom_min,self.zoom_max)
                self.fig_plot.set_xlim(Start_freq/1e6,Stop_freq/1e6)
                self.zoom_trigger = False 
                max_plot, = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[1], color=color_data_set[0])
                min_plot, = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[2], color=color_data_set[1])
                mean_plot, = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[3], color=color_data_set[2])
                self.fig_plot.set_ylabel(yaxis_label)#("Electrical Field Strength [dBuV/m]")#('Power [dBm]')
                self.fig_plot.set_xlabel("Frequency (MHz)" )#(resolution %.3f kHz)"%1)
                maxleg = 'max'
                minleg = 'min'
                meanleg = 'mean'
                #self.fig_plot.legend([max_plot,min_plot,mean_plot],['max', 'min', 'mean'])
            elif self.zoom_trigger2 == True:
                self.fig_plot.set_ylim(self.zoom_min,self.zoom_max)
                self.fig_plot.set_xlim(Start_freq/1e6,Stop_freq/1e6)
                self.zoom_trigger = False 
                max_plot, = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[1], color=color_data_set[0])
                min_plot, = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[2], color=color_data_set[1])
                mean_plot, = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[3], color=color_data_set[2])
                self.fig_plot.set_ylabel(yaxis_label)#("Electrical Field Strength [dBuV/m]")#('Power [dBm]')
                self.fig_plot.set_xlabel("Frequency (MHz)" )#(resolution %.3f kHz)"%1)
                maxleg = 'max data2'
                minleg = 'min data2'
                meanleg = 'mean data2'
                
            else:
                self.fig_plot.set_xlim(Start_freq/1e6,Stop_freq/1e6)
                self.fig_plot.set_ylim(np.min(reduced_data[2])-30,np.max(reduced_data[1])+30)
                max_plot, = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[1], color=color_data_set[0])
                min_plot, = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[2], color=color_data_set[1])
                mean_plot, = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[3], color=color_data_set[2])
                self.fig_plot.set_ylabel(yaxis_label)#("Electrical Field Strength [dBuV/m]")#('Power [dBm]')
                self.fig_plot.set_xlabel("Frequency (MHz)" )#(resolution %.3f kHz)"%1)
                #self.fig_plot.legend([max_plot,min_plot,mean_plot],['max', 'min', 'mean'])
                maxleg = 'max'
                minleg = 'min'
                meanleg = 'mean'
                
            self.fig_plot.legend([max_plot,min_plot,mean_plot],[maxleg, minleg, meanleg])
            self.canvas.draw()       
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