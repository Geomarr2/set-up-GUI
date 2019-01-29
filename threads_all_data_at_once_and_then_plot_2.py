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
from tkinter import ttk
from tkinter import scrolledtext
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
import reduceData 
import getCCFandG_LNA
from numba import vectorize
import math
from functools import reduce
#from tkinter import Label, Button, Radiobutton, IntVar

maxleg = 'max'
minleg = 'min'

PATHCCF = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/Spectrum/CCF.csv"
PATHGAINCIRCUIT = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/RFIChamberMeasure/RFcable/GainCircuitNew.csv"

fields_zoom = "Start Frequency (MHz): ", "Stop Frequency (MHz): ","Maximum amplitude: ","Minimum amplitude: "
zoom_values = ['1000', '1150', '10', '-30']

plotDataStore = {}
headTemp = {}

headInfo = {}

color_set1 = ['b','c','m', 'r','pink','g']

dataTEST = []
CCFTEST = []
GainTEST = []

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
        
        
        self.top_frame_plot = Frame(self.root)
        self.top_frame_plot.pack(side=LEFT)
        
        self.options_frame = Frame(self.top_frame) 
        self.options_frame.pack(side=LEFT)

        
        self.input_frame = Frame(self.top_frame)
        self.input_frame.pack(side=TOP)
        
        self.buttons_frame = Frame(self.top_frame_plot)
        self.buttons_frame.pack(side=RIGHT)
                
        
        self.dir_button_frame = Frame(self.bottom_frame)
        self.dir_button_frame.pack(side=RIGHT)
        
        self.zoom_button_frame = Frame(self.top_frame)
        self.zoom_button_frame.pack(side=BOTTOM)
        self.options    = GUIOptions(self.root,self)
        

        
class GUIOptions():
    headTemp  = {}
    def __init__(self,root,parent):
        
        self.root = root
        self.parent = parent
        self.IntializePlot()
        self.plotNumber = 'Load data set  '
        self.NumberOfPlot = 0

        self.directory = []
        self.head = {}
        
        # set counters
        self.missingData = 0
        self.UserInputMissingData = 0
        self.max_plot = []
        self.min_plot = []


        #create headerfile
        
        self.Filename = []
        
        self.zoom_Start_freq=0
        self.zoom_Stop_freq=0
        self.zoom_max=0
        self.zoom_min=0
        self.zoom_trigger = False
        self.original = False

        mygreen = "#d2ffd2"
        myred = "#dd0202"
        style = ttk.Style()

        style.theme_create( "yummy", parent="alt", settings={
        "TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0] } },
        "TNotebook.Tab": {
            "configure": {"padding": [5, 1], "background": mygreen },
            "map":       {"background": [("selected", myred)],
                          "expand": [("selected", [1, 1, 1, 0])] } } } )
        style.theme_use("yummy")
        self.tab_control = ttk.Notebook(self.parent.buttons_frame,height=200, width=250)
        self.tab_control.pack(side=TOP, anchor=W, pady=90)
        
        self.newRWB_entry_data = self.new_res_BW()
         
        # button to get new BW of the display resolution 
        
        self.newRBW_opts_frame = Frame(self.parent.zoom_button_frame)
        
        self.newRBW_opts_frame.pack(side=RIGHT,fill=BOTH,expand=1)

        self.show_newRBW_button=self._custom_button(
                self.newRBW_opts_frame,text = 'Replot ', 
                command=(lambda self=self:self.zoomActivate(self.plotNumber, color_set1)), 
                **DEFAULT_STYLE_2).pack(side=LEFT,anchor=SW, expand=1)  
        
  
        self.clear_opts_frame = Frame(self.parent.buttons_frame,pady=1)
        self.clear_opts_frame.pack(side=BOTTOM)
        self.clear_button=self._custom_button(
                self.clear_opts_frame,text = 'Clear Plot', 
                command=(lambda self=self: self.clear_plot_data()),
                **DEFAULT_STYLE_2)
        
        
        self.misc_opts_frame = Frame(self.parent.buttons_frame,pady=1)
        self.misc_opts_frame.pack(side=BOTTOM)
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
        
        
        
        self.submit_button = Button(self.bottom_frame,text=self.plotNumber,width=60,
                                       command=lambda self=self:self.dump_filenames(self.plotNumber, color_set1),
                                       **DEFAULT_STYLE_2).pack(side=LEFT,anchor=SW) 
  
    def GetCF(self,headerFile, plot_num):
#Load in all the header file get the info and the BW from max and min center freq
        
        file = [value for counter, value in enumerate(headerFile)]
        headerInfo = [open(headerFile[count],'r').readlines() for count, val in enumerate(headerFile)]
        temp = [value[Cnt+1] for counter, value in enumerate(headerInfo) for Cnt, Val in enumerate(value) if Val == 'Center Frequency in Hz:\n']
        temp = [float(value[:-1]) for counter, value in enumerate(temp) if value.endswith('\n')]
        temp = np.array(temp).astype('float32')
        
        return temp
    
    def checkShameHeader_Data(self,HeaderFile,dataFile):
#Load in all the header file get the info and the BW from max and min center freq
        if len(HeaderFile) != len(dataFile): 
            tkMessageBox.showwarning('Warning','Data is missing')
       # return boolTemp
    
    def printHeaderInfoNEW(self,dataFile,headerFile, plot_num, Cfreq):
#Load in all the header file get the info and the BW from max and min center freq
       # d = 
      #  headTemp = {}
        headTemp = mpifrRFIHeader.HeaderInformation()
        #print(self.head)
        #if len(headerFile) == len(dataFile):
            #temp = [np.load(value) for counter, value in enumerate(self.dataFile) for counterHeader, valueHeader in enumerate(temp) if value == (valueHeader+'.npy')]
            
        file2 = open(headerFile, "r")
        headerInfo = file2.readlines()
            
        if headerInfo[6] != 'default\n' and headerInfo[8] != 'default\n' and headerInfo[18] != 'default\n' and headerInfo[20] != 'default\n'and headerInfo[22] != 'default\n'and headerInfo[24] != 'default\n':
            headTemp.centerFrequency = float(headerInfo[4])
            headTemp.bandwidth = round(float(headerInfo[6])/10e6)*10e6
            headTemp.chambercalib = headerInfo[18]
            headTemp.antennacalib = float(headerInfo[20])
            headTemp.cablecalib = float(headerInfo[22])
            headTemp.lnaCalib = headerInfo[24]
            headTemp.nSample = int(headerInfo[8])-1
            
        else:
            headTemp.bandwidth = 40*1e6
            headTemp.chambercalib = PATHCCF 
            headTemp.antennacalib = 0.75
            headTemp.cablecalib = 1
            headTemp.lnaCalib = PATHGAINCIRCUIT
            headTemp.nSample = 373851-1
        self.CCFdata = self.CCF_DatafileCSV(headTemp.chambercalib)  
        self.GainCircuitData = self.cal_GainCircuit(headTemp.lnaCalib,max(self.CCFdata[:, 0]), min(self.CCFdata[:, 0]),plot_num, headTemp.centerFrequency,5900)#self.GainLNA_DatafileCSV(headTemp.lnaCalib)
        headTemp.dataFile = dataFile
        headTemp.headerFile = headerFile
        file2.close()
      #  else:
       #     tkMessageBox.showwarning('Warning','Data is missing')
        return headTemp
    
    def getDataID(self,headerFile):
#Load in all the header file get the info and the BW from max and min center freq
        
        headerInfo = [open(headerFile[count],'r').readlines() for count, val in enumerate(headerFile)]
   
        tempID = [value[Cnt+1] for counter, value in enumerate(headerInfo) for Cnt, Val in enumerate(value) if Val == 'Unique Scan ID:\n']
        tempID = [value[:-1] for counter, value in enumerate(tempID) if value.endswith('\n')]
        
        return tempID
        
    def dump_filenames(self, plot_num, color):
#Load and check if the datafiles correspond to the headerfiles
        # sort the file as well
        dataFile = []
        headerFile = []    
   #     dataFileTEST = []
        cnt = self.NumberOfPlot
        color = [color[cnt], color[1+cnt]]
        cnt = self.NumberOfPlot+1
        self.NumberOfPlot = cnt
        plot_num = self.plotNumber[:-1]+str(self.NumberOfPlot)
        self.plotNumber = plot_num
   #     print(self.zoom_trigger)
        for i in range(len(self.Filename)): 
            if self.Filename[i].endswith(".rfi"):
                headerFile.append(self.Filename[i])
        
        Cfreq = self.GetCF(headerFile, plot_num)
        
        Cfreq, headerFile = self.sortList(Cfreq, headerFile)
        
        tempID2 = self.getDataID(headerFile)
        
        dataFile = [self.directory+'/'+str(ID)+'.npy' for cnt, ID in enumerate(tempID2)]
        t0 = time.time()
        self.checkShameHeader_Data(dataFile,headerFile)
        self.saveHeaderFile(dataFile, headerFile, plot_num, Cfreq)

        headTemp.update(headTemp)
        self.plotDumpData(dataFile, headerFile, plot_num, Cfreq,color)
        self.displayHeaderFile(dataFile, headerFile, plot_num, Cfreq)
        print(time.time()-t0)
        
    def saveHeaderFile(self, dataFile, headerFile, plot_num, Cfreq):
        # store data in object

        self.head[plot_num] = {}
        for cnt in range(len(Cfreq)):
            self.head[plot_num][Cfreq[cnt]] = self.printHeaderInfoNEW(dataFile[cnt],headerFile[cnt], plot_num, Cfreq[cnt])   
        
    def displayHeaderFile(self, dataFile, headerFile, plot_num, Cfreq):
        # display headerinfo in tab
        plot = Frame(self.tab_control)
        self.textArea = scrolledtext.ScrolledText(plot, height=10, width=50, wrap=WORD)
        #plot2 = Frame(self.tab_control)
        #self.textArea2 = scrolledtext.ScrolledText(plot2, height=10, width=50, wrap=WORD)
        #scr = Scrollbar(tab1, command=self.textArea.yview)
        
        self.textArea.focus_set()
        self.textArea.pack(side=LEFT, fill=Y)
        
        #self.textArea2.focus_set()
        #self.textArea2.pack(side=LEFT, fill=Y)
        
        self.tab_control.add(plot,text=plot_num)
        #self.tab_control.add(plot2,text=plotNumber[1])
        
        file2 = open(headerFile[0], "r")
        self.textArea.delete('1.0', END)
        headerInfo = file2.readlines()
        [self.textArea.insert(END, i) for i in headerInfo]
        # another way of displaying info
        # tkMessageBox.showinfo(title = "Acquisition information", message = msg)   
        
    def plotDumpData(self, dataFile, headerFile, plot_num, Cfreq,color):
        # plot data using thread
        bw = self.head[plot_num][Cfreq[0]].bandwidth
        nSample = self.head[plot_num][Cfreq[0]].nSample
        Start_freq = min(Cfreq) - bw/2
        Stop_freq = max(Cfreq) + bw/2
        self.xlim_Start_freq = Start_freq/1e6
        self.xlim_Stop_freq = Stop_freq/1e6
        
        # this is used for ZOOM
        plotDataStore[plot_num] = dataFile, headerFile, Cfreq, bw, nSample
        
        dataFile = [dataFile[i] for i in range(len(dataFile)) if Cfreq[i] <= 6000*1e6 and Cfreq[i] >= 100*1e6]
        #Cfreq2 = [Cfreq[i] for i in range(len(Cfreq)) if Cfreq[i] <= 6000*1e6 and Cfreq[i] >= 100*1e6]
        
        fact = [i for i in range(1, nSample + 1) if nSample % i == 0]
        factors = sorted(i for i in fact if i >= 40)
        scaling_factor = factors[0]
        threads = []
        testData = []
        for cnt, Cf in enumerate(Cfreq):
            
            #data = self.head.readFromFile(headerFile[cnt])
            original_data = self.loadDataFromFileOLD(dataFile[cnt])
            testData.append(original_data)
            thread = reduceData.ReduceData(original_data,Cf, plot_num, scaling_factor, bw)
            thread.start()
            thread.read_reduce_Data()
            threads.append(thread.FreqMaxMinValues[Cf])
        threads = np.array(threads, dtype='float32')
        freq = threads[:,0,:].flatten() 
        specMax = threads[:,1,:].flatten()
        specMin = threads[:,2,:].flatten()
        temp = freq, specMax, specMin  
        temp = np.array(temp)
        self.calibrateDataNEW(temp,Cfreq, Stop_freq, Start_freq, color, scaling_factor, plot_num)
        self.fig_plot.set_ylim(-50, 100)
        self.fig_plot.set_xlim(self.xlim_Start_freq,self.xlim_Stop_freq)
        self.canvas.draw()  
        
    def zoom_dump_data(self, scaling_factor,color, dataFile, Cfreq, plot_num):
        # original data start frequency and stop frequency and datafile
        bw = self.head[plot_num][Cfreq[0]].bandwidth
        zoom_Start_freq_plot = self.zoom_Start_freq - bw
        zoom_Stop_freq_plot = self.zoom_Stop_freq + bw
        
        dataFile = [dataFile[i] for i in range(len(dataFile)) if Cfreq[i] <= 6000*1e6 and Cfreq[i] >= 100*1e6]
        te = [Cfreq[i] for i in range(len(Cfreq)) if Cfreq[i] <= 6000*1e6 and Cfreq[i] >= 100*1e6] 
        threads = []
        thread = 0
        new_Cfreq = np.array([], dtype = 'float32')
        t0 = time.time()
        
        for cnt, CfreqVal in enumerate(te):
            if CfreqVal >= zoom_Start_freq_plot and CfreqVal <= zoom_Stop_freq_plot:
            #data = self.head.readFromFile(headerFile[cnt])
                new_Cfreq = np.append(new_Cfreq, CfreqVal) 
                original_data = self.loadDataFromFileOLD(dataFile[cnt])
                print(original_data)
                thread = reduceData.ReduceData(original_data,CfreqVal, plot_num, scaling_factor, bw)
                thread.start()
                thread.read_reduce_Data()
                threads.append(thread.FreqMaxMinValues[CfreqVal])
        
        print(time.time()-t0)
        threads = np.array(threads, dtype='float32')
        freq = threads[:,0,:].flatten() 
        specMax = threads[:,1,:].flatten()
        specMin = threads[:,2,:].flatten()
        temp = freq, specMax, specMin  
        temp = np.array(temp)
        self.calibrateDataNEW(temp,new_Cfreq, zoom_Stop_freq_plot, zoom_Start_freq_plot, color, scaling_factor, plot_num)

        self.fig_plot.set_ylim(self.zoom_min,self.zoom_max)
        self.fig_plot.set_xlim(self.zoom_Start_freq/1e6,self.zoom_Stop_freq/1e6)
        self.canvas.draw()   
       
    def zoom_dump_data_org(self, color,headerFile, dataFile, CFFreq, nSample, plot_num):
        # original data start frequency and stop frequency and datafile

       # scaling_factor = int(self.head[plot_num][CFFreq[0]].nSample)
        original_data = []# np.array([], dtype='float32')
        d = []
        counter = []
        # I want to over sample
        bw = round(self.head[plot_num][CFFreq[0]].bandwidth/10e6)*10e6
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
            self.calibrateDataNEW(original_data,temp[0], temp[0]+bw/2, temp[0]-bw/2, color, nSample, plot_num)
    
            original_data = self.loadDataFromFile_org(d[1], temp[1],plot_num)
            self.calibrateDataNEW(original_data,temp[1], temp[1]+bw/2, temp[1]-bw/2, color, nSample, plot_num)
        else:   
            
            original_data = self.loadDataFromFile_org(d[0], temp[0],plot_num)
            self.calibrateDataNEW(original_data,temp[0], temp[0]+bw/2, temp[0]-bw/2, color, nSample, plot_num)
    
            original_data = self.loadDataFromFile_org(d[1], temp[1],plot_num)
            self.calibrateDataNEW(original_data,temp[1], temp[1]+bw/2, temp[1]-bw/2, color, nSample, plot_num)
            
            original_data = self.loadDataFromFile_org(d[2], temp[2],plot_num)
            self.calibrateDataNEW(original_data,temp[2], temp[2]+bw/2, temp[2]-bw/2, color, nSample, plot_num)
        
        self.fig_plot.set_ylim(self.zoom_min,self.zoom_max)
        self.fig_plot.set_xlim(self.zoom_Start_freq/1e6,self.zoom_Stop_freq/1e6)
        
        self.original = False
       # self.fig_plot.legend(['Original data'])
        self.canvas.draw()          

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
    
    
    # for the defined frequency range
    def get_CCF(self, Stop_freq, Start_freq): 
       CCFdata = self.CCFdata
       #temp_spec = np.array([],dtype='float32')
       
       CCFTemp = [CCFdata[cnt,1] for cnt, val in enumerate(CCFdata[:,0]) if val >= Start_freq and val < Stop_freq]
       CCFFreq = [CCFdata[cnt,0] for cnt, val in enumerate(CCFdata[:,0]) if CCFdata[cnt,0] >= Start_freq and CCFdata[cnt,0] < Stop_freq]
       
       temp_spec = [CCFFreq, CCFTemp]
       temp = np.array(temp_spec, dtype='float32')
       return temp
       '''
    def GainLNA_DatafileCSV(self,path):
        data = np.array([])  
        #temp_data = np.array([]) 
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
        #temp = [freq,dataG]
        GainLNA = self.rescale(freq, dataG) # equate CCF and gain same length
        #GainLNA = np.array(temp).astype('float32')
        return GainLNA   
        '''
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
    
    def rescale(self, Temp, Freq):
       x = np.linspace(100*1e6,6*1e9, 5900)
       temp_spec = np.interp(x, Freq, Temp)
       temp_spec2 = [x, temp_spec]
       temp = np.array(temp_spec2, dtype='float32')
       return temp
   
    def cal_GainCircuit(self, file, upperfreq, lowerfreq,plot_num, Cfreq,scaling_factor = 5900): 
        # call circuitry gain here because you dont know what wiould be used for the second measurement
       Gaintemp = []
       self.GainCircuitData = self.GainLNA_DatafileCSV(file)

       freqGain = (self.GainCircuitData[0,:])
       newfreqGain = np.linspace(lowerfreq,upperfreq,scaling_factor)
       testGain = np.interp(newfreqGain, freqGain, self.GainCircuitData[1,:])
       y = newfreqGain, testGain
       Gaintemp = np.resize(y, (2,len(testGain))).astype('float32')
       return Gaintemp

    def loadDataFromFileOLD(self,dataFile):
        spec = np.load(dataFile)

        return np.array(spec, dtype=np.float32) 

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
   
    def zoomActivate(self, plot_num, color):
        t0 = time.time()
     #   if self.zoom_trigger == False:
     #       self.clear_plot()
        plot_num = self.plotNumber
        dataFile = plotDataStore[plot_num][0]
        
        headerFile = plotDataStore[plot_num][1]
        CFreq = plotDataStore[plot_num][2]
        bw = plotDataStore[plot_num][3]
        nSample = plotDataStore[plot_num][4]

        self.zoom_trigger = True
        self.getZoomInput()
        fact = [i for i in range(1, nSample + 1) if nSample % i == 0]
        factors = sorted(i for i in fact if i >= 40)
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
            elif zoom_Bandwidth <= bw+100:
                self.original = True
                self.zoom_dump_data_org(color[2],headerFile, dataFile, CFreq, nSample, plot_num)
                
        else: 
            tkMessageBox.showwarning(title="Warning", message="Maximum zoom bandwidth must be more than acquisition bandwidth which is %d MHz."%(int(round(self.head[plot_num].bandwidth/1e6))))
        t1 = time.time()
        print(t1 - t0)
        
    def IntializePlot(self):
        self.figure = Figure(figsize=(17,10))
        self.fig_plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent.top_frame_plot)
        self.canvas.get_tk_widget().configure(background="#d469a3", highlightcolor='grey', highlightbackground='white')
        self.canvas.get_tk_widget().pack(side=BOTTOM, fill=X, padx=5, pady=5)
        self.toolbar = NavSelectToolbar(self.canvas,self.parent.top_frame_plot,self)
        self.toolbar.update()
        self.fig_plot.set_ylabel("Electrical Field Strength [dBuV/m]")#('Power [dBm]')
        self.fig_plot.set_xlabel("Frequency (MHz)" )#(resolution %.3f kHz)"%1)
        #self.cutsom_legend()
        
    def cutsom_legend(self):
        set1_max = mpatches.Patch(color=color_set1[0], label='Max data 1')
        set1_min = mpatches.Patch(color=color_set1[1], label='Min data 1')
        set2_max = mpatches.Patch(color=color_set2[0], label='Max data 2')
        set2_min = mpatches.Patch(color=color_set2[1], label='Min data 2')
        org1 = mpatches.Patch(color=color_set1[2], label='Original data 1') 
        org2 = mpatches.Patch(color=color_set2[2], label='Original data 2') 

        # Put a legend to the right of the current axis
       # self.fig_plot.legend(handles=[set1_max, set1_min, set2_max, set2_min, org1, org2], loc='upper center', bbox_to_anchor=(0.5, 1.1),
        #  fancybox=True, shadow=True, ncol=6)
    
    
    def clear_plot(self):
        #self.clear_data()
        #self.clear_plot()
        #self.canvas.clear()
        self.canvas.get_tk_widget().destroy()
        self.toolbar.destroy()
        self.canvas = None
        self.IntializePlot()
        #figure = Figure(figsize=(20,20))

        
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
            row = Frame(self.parent.zoom_button_frame)
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
       #  temp = []
          
         temp = -CCF-G_LNA + Lcable - (10.0 * np.log10(antennaEfficiency)) + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r*r)))) + 90.0

       #  temp = [(-CCF[1, cnt]-G_LNA[1, cnt] + Lcable - (10.0 * np.log10(antennaEfficiency)) + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r*r)))) + 90.0) for cntMain, valMain in enumerate(spectrum[0, :]) for cnt, val in enumerate(CCF[0, :]) if valMain < (val + 1e6) and valMain >= val]
       #  print(len(temp))
      #   temp = temp[:-abs(len(spectrum[0])-len(temp))]
      #   print(len(spectrum[0]))
         return spectrum[0], spectrum[1]+temp, spectrum[2]+temp#, spectrum[3]+temp   
     
    def calCCFdBuvPerM_Original(self,spectrum, CCF, Lcable, G_LNA, antennaEfficiency): # returns in [dBuV/m]
         # spectrum numpy array
         #get the 1MHz average of the spectrum
         Z0 = 377  # Impedance of freespace
         r = 1.0 # Distance DUT to Antenna
         antennaEfficiency = 0.75
         
         temp = -CCF-G_LNA + Lcable - (10.0 * np.log10(antennaEfficiency)) + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r*r)))) + 90.0
         return spectrum[0], spectrum[1]+temp 

    def trim(self, data, Stop_freq, Start_freq):
        Temp = [data[1, cnt] for cnt, val in enumerate(data[0, :]) if val >= Start_freq and val <= Stop_freq]
        Freq = [data[0, cnt] for cnt, val in enumerate(data[0, :]) if data[0, cnt] >= Start_freq and data[0, cnt] <= Stop_freq]
        temp_spec = [Freq, Temp]
        temp = np.array(temp_spec, dtype = 'float32')
        return temp
    
    def calibrateDataNEW(self, reduced_data, Cfreq, Stop_freq, Start_freq, color_data_set, scaling_factor, plot_num):
       
     #  nSample = self.head[plot_num][Cfreq[0]].nSample
    #   scaling_factor = len(Cfreq)*nSample
       CCF = self.get_CCF(Stop_freq, Start_freq)
       
       G_lna = self.GainCircuitData#self.cal_GainCircuit(Stop_freq, Start_freq,plot_num, Cfreq[0],scaling_factor)

       ylabel = "Electrical Field Strength [dBuV/m]"
       
       trimCCF = self.trim(CCF, Stop_freq, Start_freq)
       trimGlna = self.trim(G_lna, Stop_freq, Start_freq)
       CCF = trimCCF#np.resize(trimCCF, (2, len(reduced_data[0])))
       G_LNA = trimGlna#np.resize(trimGlna, (2, len(reduced_data[0])))
       
     #  for cntMain, valMain in enumerate(trimCCF[0, :]): 
     #      for i, val in enumerate(reduced_data[0, :]):

       if self.original == True:
           Spec = self.calCCFdBuvPerM_Original(reduced_data, CCF, self.head[plot_num][Cfreq].cablecalib, G_LNA, self.head[plot_num][Cfreq].antennacalib)
         
           self.plot_data(Spec,ylabel,color_data_set, Start_freq, Stop_freq)
           #self.original = False
       else:
          # Spec = self.calCCFdBuvPerM(reduced_data, CCF, self.head[plot_num][Cfreq[0]].cablecalib, G_LNA, self.head[plot_num][Cfreq[0]].antennacalib)
           for cntMain, valMain in enumerate(reduced_data[0, :]):  
               
              for cnt, val in enumerate(CCF[0, :]): 
           #        if valMain < (val + 1e6) and valMain >= val:
        #  temp= [(-CCF[1, cnt]-G_LNA[1, cnt] + Lcable - (10.0 * np.log10(antennaEfficiency)) + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r*r)))) + 90.0) for cntMain, valMain in enumerate(spectrum[0, :]) for cnt, val in enumerate(CCF[0, :]) if valMain < (val + 1e6) and valMain >= val]
           #temp = temp[:-abs(len(spectrum[0])-len(temp))]
             #          print(cntMain)
                 self.plot_data(self.calCCFdBuvPerM(reduced_data[:,cntMain], CCF[1, cnt], self.head[plot_num][Cfreq[0]].cablecalib, G_LNA[1,cnt], self.head[plot_num][Cfreq[0]].antennacalib),ylabel,color_data_set, Start_freq, Stop_freq)
       

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

    def plot_data(self,reduced_data,yaxis_label,color, Start_freq, Stop_freq):
        
        if self.original == True:
            self.org_data = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[1], color=color)
            self.leg = ['Original']
        else:
            self.max_plot, =self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[1], color=color[0])
            self.min_plot, = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[2], color=color[1])
            

if __name__ == '__main__':
   root = Tk()
   root.tk_setPalette(**DEFAULT_PALETTE)
   start = GUI_set_up(root)
   

   root.mainloop() 