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
import mpifrRFIHeader
from tempfile import TemporaryFile
import scipy.io
import traceback
from matplotlib.widgets import Lasso
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import queue
from numba import vectorize
import threading 
from functools import reduce
#from tkinter import Label, Button, Radiobutton, IntVar
maxleg = 'max'
minleg = 'min'
#colorbg= "#d469a3"
#FileCCF="CCF4.csv"
PATHCCF = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/Spectrum/CCF4.csv"
PATHGAINCIRCUIT = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/RFIChamberMeasure/RFcable/GainCircuitNew.csv"
#FileGainCircuit = "GainCircuit.csv"
#fields = 'Path', 'Filename', 'Start Frequency (MHz)', 'Stop Frequency (MHz)', 'LNA gain (dB)', 'Cable losses (dB)','Antenna efficiency'
fields_zoom = "Start Frequency (MHz): ", "Stop Frequency (MHz): ","Maximum amplitude: ","Minimum amplitude: "
zoom_values = ['1000', '1150', '10', '-30']
#AcqBW = 40e6
plotDataStore = {}
plotNumber = 'Load data set one', 'Load data set two' 

test = 0

color_set1 = ['b','c','m']
color_set2 = ['r','pink','g']

CCFFile = TemporaryFile()
CCFtemp = []
Gaintemp = []
testdataset1 = []

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
DEFAULT_DIRECTORY = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/"#os.getcwd()

#----------Style options---------#
DEFAULT_PALETTE = {"foreground":"blue","background":"grey"}
DEFAULT_PALETTE_NEW = {"foreground":"blue","background":"gray90"}
DEFAULT_STYLE_1 = {"foreground":"black","background":"lightblue"}
DEFAULT_STYLE_2 = {"foreground":"gray90","background":"darkgreen"}
DEFAULT_STYLE_3 = {"foreground":"gray90","background":"darkred"}
DEFAULT_STYLE_NEW = {"foreground":"gray90","background":"darkgreen"}
figure = Figure(figsize=(10,10))
fig_plot = figure.add_subplot(111)


continuePlotting = False
 
def change_state():
    global continuePlotting
    if continuePlotting == True:
        continuePlotting = False
    else:
        continuePlotting = True

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

class GUI_set_up():
    def __init__(self, root):   
        
        self.root = root
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
        
class ReduceData(threading.Thread):
    FreqMaxMinValues = {}
    MinValues = {}
    #lock = threading.Lock()
    #q = queue.Queue()
    def __init__(self,original_data,Cfreq, plot_num, scaling_factor, bw, CCFdata, color, Lcable, LNAdata, ANTdata):
        #threading.Thread.__init__(self)
        self.ANTdata = ANTdata
        self.CCFdata = CCFdata
        self.LNAdata = LNAdata
        self.color = color
        self.Lcable = Lcable
        self.original_data = original_data
        self.Cfreq = Cfreq   
        self.plot_num = plot_num
        self.scaling_factor = scaling_factor
        self.bw = bw
        #ReduceData.lock.acquire()
        #ReduceData.FreqMaxMinValues[Cfreq] = []
        #ReduceData.lock.release()
        
        self.calibrateData()

    def read_reduce_Data(self):
        # read in the whole BW in one array
# set the display sample size depending on the display bandwidth and resolution  
        spec = self.dBuV_M2V_M(self.original_data)
        x = int(len(self.original_data)/self.scaling_factor)
        spec_min = np.array([], dtype=np.float32)
        spec_max = np.array([], dtype=np.float32)
        freq = np.array([], dtype=np.float32)
        Start_freq = self.Cfreq -self.bw/2
        Stop_freq = self.Cfreq +self.bw/2
        spec_max = [np.max(spec[(i*x):(x*i+x)]) for i in range (self.scaling_factor)]
        spec_min = [np.min(spec[(i*x):(x*i+x)]) for i in range (self.scaling_factor)]
        spec_max = self.V_M2dBuV_M(spec_max)
        spec_min = self.V_M2dBuV_M(spec_min)
        freq = np.linspace(Start_freq,Stop_freq,len(spec_max)) 
        temp = freq,spec_max,spec_min
        

        return temp
    def dBuV_M2V_M(self,spec):
        VperM = pow(10,(spec-120)/20)
        return VperM    
    
    def V_M2dBuV_M(self,spec):
        dBuV_M = 20*np.log10(spec)+120
        return dBuV_M  
    
    def calibrateData(self):
       reduced_data = self.read_reduce_Data()
       CCF = self.get_CCF()
       G_LNA = self.cal_GainCircuit()

       
       if self.original == True:
           Spec = self.calCCFdBuvPerM_Original(reduced_data, CCF, self.Lcable, G_LNA, self.ANTdata)
           self.plot_data(Spec)
           #self.original = False
       else:
           ReduceData.lock.acquire()
           
           self.plot_data(self.calCCFdBuvPerM(reduced_data, CCF, self.Lcable, G_LNA, self.ANTdata))
           ReduceData.lock.release()
           
           
           

       #return data   
   
    def plot_data(self,reduced_data):
        if self.original == True:
            self.org_data = fig_plot.plot(reduced_data[0]/1e6,reduced_data[1], color=self.color)
            self.leg = ['Original']
        else:
            self.max_plot, = fig_plot.plot(reduced_data[0]/1e6,reduced_data[1], color=self.color[0])
            self.min_plot, = fig_plot.plot(reduced_data[0]/1e6,reduced_data[2], color=self.color[1])
    
    def cal_GainCircuit(self): 
        # call circuitry gain here because you dont know what wiould be used for the second measurement
       
       freqGain = (self.LNAdata[0,:])
       newfreqGain = np.linspace(Start_freq,Stop_freq,self.scaling_factor)
       testGain = np.interp(newfreqGain, freqGain, self.LNAdata[1,:])
       Gaintemp.append([newfreqGain, testGain])
       return testGain 

    def get_CCF(self): 
       CCFdata = self.CCFdata
       Start_freq = self.Cfreq -self.bw/2
       Stop_freq = self.Cfreq +self.bw/2
       temp_spec = np.array([],dtype='float32')
       
       CCFTemp = [CCFdata[cnt,1] for cnt, val in enumerate(CCFdata[:,0]) if val >= Start_freq and val <= Stop_freq]
       CCFFreq = [CCFdata[cnt,0] for cnt, val in enumerate(CCFdata[:,0]) if CCFdata[cnt,0] >= Start_freq and CCFdata[cnt,0] <= Stop_freq]
       
       x = np.linspace(Start_freq, Stop_freq, self.scaling_factor)
       temp_spec = np.interp(x, CCFFreq, CCFTemp)
       return temp_spec     
   
class GUIOptions():
    q = queue.Queue()
    def __init__(self,root,parent):
        threading.Thread.__init__(self)
        self.root = root
        self.parent = parent
        self.IntializePlot()
        #threading.Thread(target=lambda self=self:self.dump_filenames(plotNumber[1], color_set2)).start() 
        

        self.directory = []
        self.head = {}
        
        # set counters
        self.missingData = 0
        self.UserInputMissingData = 0
        self.max_plot = []
        self.min_plot = []
        
        #create headerfile
        self.head[plotNumber[0]] = mpifrRFIHeader.HeaderInformation()
        self.head[plotNumber[1]] = mpifrRFIHeader.HeaderInformation()
        # load CCF now not need to load every time you get the CCF of every 40MHz chunck
        
        #self.dataFile = []
        self.Filename = []
        #self.headerInfo = []
       # self.CFreq = 0
        #self.CFreq2 = 0
        
        
        self.zoom_Start_freq=0
        self.zoom_Stop_freq=0
        self.zoom_max=0
        self.zoom_min=0
        self.zoom_trigger = False
        self.original = False
        
        self.mode_select_frame = Frame(self.root,pady=20)
        self.mode_select_frame.pack(side=TOP,fill=X,expand=1)
        self.ax_opts_frame = Frame(self.root)
        self.ax_opts_frame.pack(side=TOP,expand=1)
        self.view_toggle_frame = Frame(self.root,pady=20)
        self.view_toggle_frame.pack(side=TOP,fill=X,expand=1)
        
        self.newRWB_entry_data = self.new_res_BW()
        
        
        # button to get new BW of the display resolution 
        
        self.newRBW_opts_frame = Frame(self.parent.newRBW_button_frame)
        
        self.newRBW_opts_frame.pack(side=RIGHT,fill=BOTH,expand=1)
        '''
        self.show_newRBW_button=self._custom_button(
                self.newRBW_opts_frame,text = 'Replot Second Data', 
                command=(lambda self=self:self.zoomActivate2Data()),
                **DEFAULT_STYLE_2)
        '''
        self.show_newRBW_button=self._custom_button(
                self.newRBW_opts_frame,text = 'Replot '+plotNumber[0].lstrip('Load'), 
                command=(lambda self=self:self.zoomActivate(plotNumber[0], color_set1)), 
                **DEFAULT_STYLE_2).pack(side=LEFT,anchor=SW, expand=1)  
        
        self.show_newRBW_button=self._custom_button(
                self.newRBW_opts_frame,text = 'Replot '+plotNumber[1].lstrip('Load'), 
                command=(lambda self=self:self.zoomActivate(plotNumber[1], color_set2)), 
                **DEFAULT_STYLE_2).pack(side=RIGHT,anchor=SE, expand=1) 

        self.clear_opts_frame = Frame(self.parent.buttons_frame,pady=1)
        self.clear_opts_frame.pack(side=TOP,fill=X,expand=1)
        self.clear_button=self._custom_button(
                self.clear_opts_frame,text = 'Clear Plot', 
                command=(lambda self=self: self.clear_plot_data()),
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
        
        
        
        self.submit_button = Button(self.bottom_frame,text=plotNumber[0],width=60,
                                       command=lambda self=self:self.dump_filenames(plotNumber[0], color_set1),
                                       **DEFAULT_STYLE_2).pack(side=LEFT,anchor=SW) 
        

        self.submit_button = Button(self.bottom_frame,text=plotNumber[1],width=60,
                                       command=lambda self=self:self.dump_filenames(plotNumber[1], color_set2),
                                       **DEFAULT_STYLE_2).pack(side=RIGHT,anchor=SE)

        
    def dump_filenames(self, plot_num, color):
#Load in vereything and seperate header and data file
        dataFile = []
        headerFile = []    
        dataFileTEST = []
   #     print(self.zoom_trigger)
        for i in range(len(self.Filename)): 
            if self.Filename[i].endswith(".rfi"):
                headerFile.append(self.Filename[i])
            elif self.Filename[i].endswith(".npy"):
                dataFileTEST.append(self.Filename[i])
        
        msg, tem = self.printHeaderInfo(dataFileTEST, headerFile, plot_num)
        
        tem, headerFile = self.sortList(tem, headerFile)
        
        tempID2 = self.head[plot_num].getDataID(headerFile)
        
        dataFile = [self.directory+'/'+str(ID)+'.npy' for cnt, ID in enumerate(tempID2)]
        
        self.head[plot_num].centerFrequency = tem
        self.head[plot_num].bandwidth = float(msg[6])
        bw = self.head[plot_num].bandwidth
        self.xlim_Start_freq = (min(tem) - bw/2)/1e6
        self.xlim_Stop_freq = (max(tem) + bw/2)/1e6
        
        self.head[plot_num].dataFile =  dataFile
        self.head[plot_num].headerFile =  headerFile
        
        plotDataStore[plot_num] = dataFile, headerFile,tem, self.head[plot_num]
        
        dataFile = [dataFile[i] for i in range(len(dataFile)) if tem[i] <= 6000*1e6 and tem[i] >= 100*1e6]
        te = [tem[i] for i in range(len(tem)) if tem[i] <= 6000*1e6 and tem[i] >= 100*1e6]
        
        
        nSample = self.head[plot_num].nSample
        fact = [i for i in range(1, nSample + 1) if nSample % i == 0]
        factors = sorted(i for i in fact if i >= 40)
        scaling_factor = factors[0]

        t0 = time.time()
        for cnt, Cfreq in enumerate(te):
            original_data = self.loadDataFromFileOLD(dataFile[cnt])
    # Event for stopping the IO thread
            run = threading.Event()
            run.set()

    # Run io function in a thread
            t = threading.Thread(target=ReduceData, args=(original_data,Cfreq, plot_num, scaling_factor, bw, self.CCFdata, self.head[plot_num].cablecalib,color, self.LNAdata, self.head[plot_num].antennacalib))
            t.start()
            
            #thread = ReduceData(original_data,Cfreq, plot_num, scaling_factor, bw, self.CCFdata, self.head[plot_num].cablecalib,color, self.LNAdata, self.head[plot_num].antennacalib)
            #thread.Thread(target=(lambda: calibrateData()
            
            #threads.append(thread.FreqMaxMinValues[Cfreq])
            #thread.start()
        print(time.time()-t0)
        
        #threads = np.array(threads, dtype='float32')
        #freq = threads[:,0,:].flatten() 
        #specMax = threads[:,1,:].flatten()
        #specMin = threads[:,2,:].flatten()
        
        
        #temp = freq, specMax, specMin  
        #temp = np.array(temp)
        
        fig_plot.set_ylim(-50, 100)
        fig_plot.set_xlim(self.xlim_Start_freq,self.xlim_Stop_freq)
        self.canvas.draw()   
        
    def printHeaderInfo(self,dataFile,headerFile, plot_num):
#Load in all the header file get the info and the BW from max and min center freq
        
        file = [value for counter, value in enumerate(headerFile)]
        headerInfo = [open(headerFile[count],'r').readlines() for count, val in enumerate(headerFile)]
        temp = [value[Cnt+1] for counter, value in enumerate(headerInfo) for Cnt, Val in enumerate(value) if Val == 'Center Frequency in Hz:\n']
        temp = [float(value[:-1]) for counter, value in enumerate(temp) if value.endswith('\n')]
        temp = np.array(temp)#.astype('float32')
        
#check if all the numpy arrays is there
        if len(temp) == len(dataFile):
            file2 = open(headerFile[0], "r")
            msg = []
            headerInfo = file2.readlines()
            if headerInfo[6] != 'default\n' and headerInfo[8] != 'default\n' and headerInfo[18] != 'default\n' and headerInfo[20] != 'default\n'and headerInfo[22] != 'default\n'and headerInfo[24] != 'default\n':
                self.head[plot_num].bandwidth = float(headerInfo[6])
                self.head[plot_num].chambercalib = headerInfo[18]
                self.head[plot_num].antennacalib = float(headerInfo[20])
                self.head[plot_num].cablecalib = float(headerInfo[22])
                self.head[plot_num].lnaCalib = headerInfo[24]
                self.head[plot_num].nSample = int(headerInfo[8])-1
            else:
                self.head[plot_num].bandwidth = 40*1e6
                self.head[plot_num].chambercalib = PATHCCF 
                self.head[plot_num].antennacalib = 0.75
                self.head[plot_num].cablecalib = 1
                self.head[plot_num].lnaCalib = PATHGAINCIRCUIT
                self.head[plot_num].nSample = 373851-1
            self.CCFdata = self.CCF_DatafileCSV(self.head[plot_num].chambercalib)    
            self.LNAdata = self.GainLNA_DatafileCSV(self.head[plot_num].lnaCalib)
            for x in headerInfo:
                msg.append(x)
            file2.close()
            tkMessageBox.showinfo(title = "Acquisition information", message = msg)
        else:
            tkMessageBox.showwarning('Warning','Data is missing')
        return msg, temp
    
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
        
    def getDataID(self,headerFile):
#Load in all the header file get the info and the BW from max and min center freq
        
        headerInfo = [open(headerFile[count],'r').readlines() for count, val in enumerate(headerFile)]
   
        tempID = [value[Cnt+1] for counter, value in enumerate(headerInfo) for Cnt, Val in enumerate(value) if Val == 'Unique Scan ID:\n']
        tempID = [value[:-1] for counter, value in enumerate(tempID) if value.endswith('\n')]
        
        return tempID

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
   # @staticmethod    
    def loadDataFromFile(self,dataFile, plot_num, scaling_factor, Cfreq, color):
        spec = np.load(dataFile)
        #freq = np.linspace(Cfreq-self.head[plot_num].bandwidth/2,Cfreq+self.head[plot_num].bandwidth/2,len(spec))
        #temp = [freq,spec]
        spec = np.array(spec, dtype=np.float32) 
        Stopfreq = Cfreq+self.head[plot_num].bandwidth/2
        Startfreq = Cfreq-self.head[plot_num].bandwidth/2
        self.calibrateData(self.read_reduce_Data(spec,Cfreq, plot_num, scaling_factor), Stopfreq, Startfreq, color, scaling_factor, plot_num)

  #      return np.array(spec, dtype=np.float32) 
    
    def loadDataFromFileOLD(self,dataFile):
        spec = np.load(dataFile)
        #freq = np.linspace(Cfreq-self.head[plot_num].bandwidth/2,Cfreq+self.head[plot_num].bandwidth/2,len(spec))
        #temp = [freq,spec]
     #    self.calibrateData(self.read_reduce_Data(original_data,value, plot_num, scaling_factor), Stopfreq, Startfreq, color, scaling_factor, plot_num)

        return np.array(spec, dtype=np.float32) 
    def loadDataFromFile_org(self,dataFile, Cfreq,plot_num):
        spec = np.load(dataFile)
        spec = spec[0:spec.size-1]
        freq = np.linspace(Cfreq-self.head[plot_num].bandwidth/2,Cfreq+self.head[plot_num].bandwidth/2,self.head[plot_num].nSample)
        temp = [freq,spec]
        
        return np.array(temp, dtype=np.float32) 
    
    def launch_dir_finder(self):
        self.directory = tkFileDialog.askdirectory()
        Files=[] #list of files
        for file in os.listdir(self.directory):
            Files.append(self.directory+'/'+file)
            
        self.Filename = Files
        
        
    def sortList(self, cfreq,headerFile):
        i=0
        while i<len(cfreq):
            key=i
            j=i+1
            while j<len(cfreq):
                if cfreq[key]>cfreq[j]:
                    key=j
                j+=1
            headerFile[i],headerFile[key]=headerFile[key],headerFile[i]
            cfreq[i],cfreq[key]=cfreq[key],cfreq[i]
            i+=1
        return cfreq,headerFile
   
    def zoom_dump_data(self, scaling_factor,color, dataFile, CFFreq, plot_num):
        # original data start frequency and stop frequency and datafile
        bw = self.head[plot_num].bandwidth
        zoom_Start_freq_plot = self.zoom_Start_freq - bw
        zoom_Stop_freq_plot = self.zoom_Stop_freq + bw
        
        dataFile = [dataFile[i] for i in range(len(dataFile)) if CFFreq[i] <= 6000*1e6 and CFFreq[i] >= 100*1e6]
        te = [CFFreq[i] for i in range(len(CFFreq)) if CFFreq[i] <= 6000*1e6 and CFFreq[i] >= 100*1e6] 
              
        for cnt, value in enumerate(te):
             if value >= zoom_Start_freq_plot and value <= zoom_Stop_freq_plot:
                 Startfreq = value -bw/2
                 Stopfreq = value + bw/2
                 originaldata = self.loadDataFromFile(dataFile[cnt])
                 
                 self.calibrateData(self.read_reduce_Data(originaldata,value,plot_num, scaling_factor), Stopfreq, Startfreq, color, scaling_factor, plot_num)

        fig_plot.set_ylim(self.zoom_min,self.zoom_max)
        fig_plot.set_xlim(self.zoom_Start_freq/1e6,self.zoom_Stop_freq/1e6)
        #self.fig_plot.legend(self.leg_data,self.leg)
        self.canvas.draw()   
        
    def zoom_dump_data_org(self, color,headerFile, dataFile, CFFreq, scaling_factor, plot_num):
        # original data start frequency and stop frequency and datafile

        scaling_factor = int(self.head[plot_num].nSample)
        original_data = []# np.array([], dtype='float32')
        d = []
        counter = []
        # I want to over sample
        bw = self.head[plot_num].bandwidth
        zoom_Start_freq_plot = self.zoom_Start_freq - bw
        zoom_Stop_freq_plot = self.zoom_Stop_freq + bw
        Start_freq = min(CFFreq) - (20*1e6)
        Stop_freq = max(CFFreq) + (20*1e6)
        dataFile = [dataFile[i] for i in range(len(dataFile)) if CFFreq[i] <= 6000*1e6 and CFFreq[i] >= 100*1e6]
        te = [CFFreq[i] for i in range(len(CFFreq)) if CFFreq[i] <= 6000*1e6 and CFFreq[i] >= 100*1e6]             
        if self.zoom_Start_freq <= Start_freq:
            zoom_Start_freq_plot = self.zoom_Start_freq 
        elif self.zoom_Stop_freq >= Stop_freq:
            zoom_Stop_freq_plot = self.zoom_Stop_freq 
            
        for cnt, value in enumerate(te):
                 if value >= zoom_Start_freq_plot and value <= zoom_Stop_freq_plot:
                     d.append(dataFile[cnt])
                     counter.append(cnt)
                     
        headerInfo = [open(headerFile[val],'r').readlines() for count, val in enumerate(counter)]
        temp = [value[Cnt+1] for counter, value in enumerate(headerInfo) for Cnt, Val in enumerate(value) if Val == 'Center Frequency in Hz:\n']
        temp = [float(value[:-1]) for counter, value in enumerate(temp) if value.endswith('\n')]
        temp = np.array(temp).astype('float32')
        if len(temp) <= 2:
            
            original_data = self.loadDataFromFile_org(d[0], temp[0],plot_num)
            self.calibrateData(original_data, temp[0]+bw/2, temp[0]-bw/2, color, scaling_factor, plot_num)
    
            original_data = self.loadDataFromFile_org(d[1], temp[1],plot_num)
            self.calibrateData(original_data, temp[1]+bw/2, temp[1]-bw/2, color, scaling_factor, plot_num)
        else:   
            
            original_data = self.loadDataFromFile_org(d[0], temp[0],plot_num)
            self.calibrateData(original_data, temp[0]+bw/2, temp[0]-bw/2, color, scaling_factor, plot_num)
    
            original_data = self.loadDataFromFile_org(d[1], temp[1],plot_num)
            self.calibrateData(original_data, temp[1]+bw/2, temp[1]-bw/2, color, scaling_factor, plot_num)
            
            original_data = self.loadDataFromFile_org(d[2], temp[2],plot_num)
            self.calibrateData(original_data, temp[2]+bw/2, temp[2]-bw/2, color, scaling_factor, plot_num)
        
        fig_plot.set_ylim(self.zoom_min,self.zoom_max)
        fig_plot.set_xlim(self.zoom_Start_freq/1e6,self.zoom_Stop_freq/1e6)
        
        self.original = False
       # self.fig_plot.legend(['Original data'])
        self.canvas.draw()   


    def zoomActivate(self, plot_num, color):
        t0 = time.time()
     #   if self.zoom_trigger == False:
     #       self.clear_plot()
        #dataFile = plotDataStore[plot_num][0]
        #headerFile = plotDataStore[plot_num][1]
        CFreq = plotDataStore[plot_num][2]
        
        nSample = getattr(plotDataStore[plot_num][3], 'nSample')
        dataFile = getattr(plotDataStore[plot_num][3], 'dataFile')
        headerFile = getattr(plotDataStore[plot_num][3], 'headerFile')
        

        self.zoom_trigger = True
      #  self.zoom_trigger1 = True
        self.getZoomInput()
        fact = [i for i in range(1, nSample + 1) if nSample % i == 0]
        factors = sorted(i for i in fact if i >= 40)
        bw = self.head[plot_num].bandwidth
        zoom_Bandwidth = self.zoom_Stop_freq - self.zoom_Start_freq
        zoom_nr_smp = int(zoom_Bandwidth/(bw))
        if zoom_Bandwidth >= 35*1e6:
            if zoom_nr_smp > 111:
                scaling_factor = factors[0]
                self.zoom_dump_data(scaling_factor, color, dataFile, CFreq,plot_num)
            elif zoom_nr_smp < 111 and zoom_nr_smp >= 74:
                scaling_factor = factors[1]
                self.zoom_dump_data(scaling_factor, color, dataFile, CFreq, plot_num)
            elif zoom_nr_smp < 74 and zoom_nr_smp >= 37:
                scaling_factor = factors[2]
                self.zoom_dump_data(scaling_factor, color, dataFile, CFreq, plot_num)
            elif zoom_nr_smp < 37 and zoom_nr_smp >= 1 and zoom_Bandwidth > 40*1e6:
                scaling_factor = factors[3]
                self.zoom_dump_data(scaling_factor, color, dataFile, CFreq, plot_num)
            elif zoom_Bandwidth <= bw+100:#zoom_nr_smp <= 1:
                #plot orginal    
                self.original = True
                scaling_factor = 200000
                # create a seperate original function
                self.zoom_dump_data_org(color[2],headerFile, dataFile, CFreq, nSample, plot_num)
                
        else: 
            tkMessageBox.showwarning(title="Warning", message="Maximum zoom bandwidth must be more than acquisition bandwidth which is %d MHz."%(int(round(self.head[plot_num].bandwidth/1e6))))
        t1 = time.time()
        print(t1 - t0)
    def IntializePlot(self):

        self.canvas = FigureCanvasTkAgg(figure, master=self.parent.top_frame_plot)
        self.canvas.get_tk_widget().configure(background="#d469a3", highlightcolor='grey', highlightbackground='white')
        self.canvas.get_tk_widget().pack(side=BOTTOM, fill=X, padx=5, pady=5)
        self.toolbar = NavSelectToolbar(self.canvas,self.parent.top_frame_plot,self)
        self.toolbar.update()
        fig_plot.set_ylabel("Electrical Field Strength [dBuV/m]")#('Power [dBm]')
        fig_plot.set_xlabel("Frequency (MHz)" )#(resolution %.3f kHz)"%1)
        self.cutsom_legend()
        
    def cutsom_legend(self):
        set1_max = mpatches.Patch(color=color_set1[0], label='Max data 1')
        set1_min = mpatches.Patch(color=color_set1[1], label='Min data 1')
        set2_max = mpatches.Patch(color=color_set2[0], label='Max data 2')
        set2_min = mpatches.Patch(color=color_set2[1], label='Min data 2')
        org1 = mpatches.Patch(color=color_set1[2], label='Original data 1') 
        org2 = mpatches.Patch(color=color_set2[2], label='Original data 2') 

        # Put a legend to the right of the current axis
        fig_plot.legend(handles=[set1_max, set1_min, set2_max, set2_min, org1, org2], loc='upper center', bbox_to_anchor=(0.5, 1.1),
          fancybox=True, shadow=True, ncol=6)
    
    def clear_plot(self):
        #self.clear_data()
        self.canvas.get_tk_widget().destroy()
        self.toolbar.destroy()
        self.canvas = None
        self.IntializePlot()

    def clear_plot_data(self):
        answer = tkMessageBox.askyesno(title="Clear Plot and Data", message="Are you sure you want to clear the loaded data?")
        if (answer):
            self.clear_data()
            self.clear_plot()
            tkMessageBox.showwarning(title="Clear Plot and Data", message="Data has been deleted.")
            
    def clear_data(self):
        self.zoom_data = None
        self.original = None
        self.zoom_trigger = None
        self.zoom_trigger1 = None
        self.zoom_trigger2 = None
        #self.clear_plot()
            
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
    
    
    def checkDir(self, dirname, autocreate = True, poperror = True):
        if dirname is None or len(dirname) == 0:
            return False
        try:
            if not os.path.exists(dirname) and autocreate:
                os.makedirs(dirname)
        except Exception as e:
            if poperror:
                messagebox.showinfo("Check Directory Failed: " + str(e),
                        traceback.format_exception(*sys.exc_info()))
            return False
        return True
        
    def _custom_button(self,root,text,command,**kwargs):
        button = Button(root, text=text,
            command=command,padx=2, pady=2,height=1, width=10,**kwargs)
        button.pack(side=TOP,fill=BOTH)
        return button
    
    def quit(self):
        msg = "Quitting:\nUnsaved progress will be lost.\nDo you wish to Continue?"
        if tkMessageBox.askokcancel("Combustible Lemon",msg):
            self.parent.root.destroy()    
    

     
    def calCCFdBuvPerM(self,spectrum, CCF, Lcable, G_LNA, antennaEfficiency): # returns in [dBuV/m]
         # spectrum numpy array
         #get the 1MHz average of the spectrum
         Z0 = 377  # Impedance of freespace
         r = 1.0 # Distance DUT to Antenna
         antennaEfficiency = 0.75
         
         temp = -CCF-G_LNA + Lcable - (10.0 * np.log10(antennaEfficiency)) + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r*r)))) + 90.0
         
         return spectrum[0], spectrum[1]+temp, spectrum[2]+temp#, spectrum[3]+temp   
     
    def calCCFdBuvPerM_Original(self,spectrum, CCF, Lcable, G_LNA, antennaEfficiency): # returns in [dBuV/m]
         # spectrum numpy array
         #get the 1MHz average of the spectrum
         Z0 = 377  # Impedance of freespace
         r = 1.0 # Distance DUT to Antenna
         antennaEfficiency = 0.75
         
         temp = -CCF-G_LNA + Lcable - (10.0 * np.log10(antennaEfficiency)) + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r*r)))) + 90.0
         return spectrum[0], spectrum[1]+temp 
     

       
  #     return Spec
  
    def calibrateDataNEW(self, reduced_data, Cfreq, Stop_freq, Start_freq, color_data_set, scaling_factor, plot_num):
       scaling_factor = len(Cfreq)*scaling_factor
       CCF = self.get_CCF(scaling_factor, Stop_freq, Start_freq,  plot_num)
       G_LNA = self.cal_GainCircuit(Stop_freq, Start_freq,plot_num, scaling_factor)

       
       ylabel = "Electrical Field Strength [dBuV/m]"
       if self.original == True:
           Spec = self.calCCFdBuvPerM_Original(reduced_data, CCF, self.head[plot_num].cablecalib, G_LNA, self.head[plot_num].antennacalib)
           self.plot_data(Spec,ylabel,color_data_set, Start_freq, Stop_freq)
           #self.original = False
       else:
           Spec = self.calCCFdBuvPerM(reduced_data, CCF, self.head[plot_num].cablecalib, G_LNA, self.head[plot_num].antennacalib)
           self.plot_data(Spec,ylabel,color_data_set, Start_freq, Stop_freq)
       
  #     return Spec
   

    '''   
    def read_reduce_Data(self,spectrum,cfreq, plot_num, scaling_factor):
        # read in the whole BW in one array
# set the display sample size depending on the display bandwidth and resolution  
     #   t0 = time.time()
        spec = self.dBuV_M2V_M(spectrum)
        #freq = spectrum[0]
        x = int(len(spectrum)/scaling_factor)
        spec_min = np.array([], dtype=np.float32)
        spec_max = np.array([], dtype=np.float32)
        freq = np.array([], dtype=np.float32)
        #freq_max = np.array([], dtype=np.float32)
        bw = self.head[plot_num].bandwidth
        Start_freq = cfreq -bw/2
        Stop_freq = cfreq +bw/2
        scale = range(scaling_factor)
#        spec_max = np.max(spec[(i*x):(x*i+x)])
        spec_max = [np.max(spec[(i*x):(x*i+x)]) for i in range (scaling_factor)]
        #ind_max  = [(np.argmax(spec[(i*x):(x*i+x)])+x*i) for i in range (scaling_factor)]
        spec_min = [np.min(spec[(i*x):(x*i+x)]) for i in range (scaling_factor)]
        #ind_min  = [(np.argmin(spec[(i*x):(x*i+x)])+x*i) for i in range (scaling_factor)]
        #freq_max = [freq[value] for count, value in enumerate(ind_max)]
        #freq_min = [freq[value] for count, value in enumerate(ind_min)]
            #self.leg_data = [spec_max, spec_min] 
        spec_max = self.V_M2dBuV_M(spec_max)
        spec_min = self.V_M2dBuV_M(spec_min)
        freq = np.linspace(Start_freq,Stop_freq,len(spec_max)) 
       
        temp = freq,spec_max,spec_min
        data = np.array(temp, dtype=np.float32)
       # print(time.time)
        return data       
    
    def dBuV_M2V_M(self,spec):
        VperM = pow(10,(spec-120)/20)
        return VperM    
    
    def V_M2dBuV_M(self,spec):
        dBuV_M = 20*np.log10(spec)+120
        return dBuV_M  
    '''
    def getZoomInput(self):
       # print(self.newRWB_entry_data["Start Frequency (MHz): "].get())
        if self.newRWB_entry_data["Start Frequency (MHz): "].get() == '':
            [self.newRWB_entry_data[val].insert(END, zoom_values[cnt]) for cnt, val in enumerate(fields_zoom)]
            self.zoom_Start_freq = float(self.newRWB_entry_data["Start Frequency (MHz): "].get())*1e6
            self.zoom_Stop_freq = float(self.newRWB_entry_data["Stop Frequency (MHz): "].get())*1e6
            self.zoom_max = float(self.newRWB_entry_data["Maximum amplitude: "].get())
            self.zoom_min = float(self.newRWB_entry_data["Minimum amplitude: "].get())
        else:
            self.zoom_Start_freq = float(self.newRWB_entry_data["Start Frequency (MHz): "].get())*1e6
            self.zoom_Stop_freq = float(self.newRWB_entry_data["Stop Frequency (MHz): "].get())*1e6
            self.zoom_max = float(self.newRWB_entry_data["Maximum amplitude: "].get())
            self.zoom_min = float(self.newRWB_entry_data["Minimum amplitude: "].get())
        '''
        if self.newRWB_entry_data["Start Frequency (MHz): "].get() == '':
            tkMessageBox.showwarning(title="Warning", message="Please enter the Start Frequency in MHz")
        elif self.newRWB_entry_data["Stop Frequency (MHz): "].get() == '':
            tkMessageBox.showwarning(title="Warning", message="Please enter the Stop Frequency in MHz")
        elif self.newRWB_entry_data["Maximum amplitude: "].get() == '':
            tkMessageBox.showwarning(title="Warning", message="Please enter the Maximum amplitude")
        elif self.newRWB_entry_data["Minimum amplitude: "].get() == '':
            tkMessageBox.showwarning(title="Warning", message="Please enter the Minimum amplitude")
        '''

        
    def plot_data(self,reduced_data,yaxis_label,color, Start_freq, Stop_freq):
        
        if self.original == True:
            self.org_data = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[1], color=color)
            self.leg = ['Original']
        else:
            self.max_plot, = fig_plot.plot(reduced_data[0]/1e6,reduced_data[1], color=color[0])
            self.min_plot, = fig_plot.plot(reduced_data[0]/1e6,reduced_data[2], color=color[1])
          #  self.canvas.update()
          #  self.canvas.flush_events()
            #self.leg = ['Max', 'Min']
            #self.leg_data = [self.max_plot, self.min_plot]
            
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