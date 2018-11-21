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
get_ipython().magic('reset -sf')
import numpy as np
from io import StringIO
global fields 
import calData as cal
import os
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

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.widgets import Lasso
from matplotlib.figure import Figure
FileCCF="CCF4.csv"
PathCCF = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/Spectrum/"
PathGainCircuit = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/RFIChamberMeasure/RFcable/"
FileGainCircuit = "GainCircuit.csv"
fields = 'Path', 'Filename', 'Start Frequency (MHz)', 'Stop Frequency (MHz)', 'LNA gain (dB)', 'Cable losses (dB)','Antenna efficiency'
fields_zoom = "Start Frequency (MHz): ", "Stop Frequency (MHz): ","Maximum amplitude: ","Minimum amplitude: "
AcqBW = 40e6
color = ['y','hotpink','olive','coral','r','b','m','c','g']

bandwidth = AcqBW #Hz
#StartFreq = 1000e6
#StopFreq = 1200e6
Z0 = 119.9169832 * np.pi  # Impedance of freespace
G_LNA= 20 #dB gain of the LNA
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
DEFAULT_DIRECTORY = os.getcwd()

#----------Style options---------#
DEFAULT_PALETTE = {"foreground":"blue","background":"lightblue"}
DEFAULT_STYLE_1 = {"foreground":"black","background":"lightblue"}
DEFAULT_STYLE_2 = {"foreground":"gray90","background":"darkgreen"}
DEFAULT_STYLE_3 = {"foreground":"gray90","background":"darkred"}

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
        self.top_frame.pack(side=RIGHT)
        
        self.top_frame_plot = Frame(self.root)
        self.top_frame_plot.pack(side=LEFT)
        self.plot_frame_toolbar = Frame(self.top_frame_plot)
        self.plot_frame_toolbar.pack(side=TOP)
        self.plot_frame = Frame(self.top_frame_plot)
        self.plot_frame.pack(side=TOP)
        
        self.input_frame = Frame(self.top_frame)
        self.input_frame.pack(side=TOP)
        self.buttons_frame = Frame(self.input_frame)
        self.buttons_frame.pack(side=RIGHT)
        self.dir_button_frame = Frame(self.top_frame)
        self.dir_button_frame.pack(side=BOTTOM)
        self.newRBW_button_frame = Frame(self.top_frame)
        self.newRBW_button_frame.pack(side=BOTTOM)
        
        self.options_frame = Frame(self.top_frame) 
        self.options_frame.pack(side=LEFT)
        self.options    = GUIOptions(self.options_frame,self)
        self.options.plot()
        
class GUIOptions(object):
    def __init__(self,root,parent):
        self.root = root
        self.parent = parent
        self.Path= "C:/Users/geomarr/Documents/GitHub/set-up-GUI/GUIData/"
        self.Filename = "Spectrum_"
        self.CenterFrequency = 1020*1e6
        self.Bandwidth = 40*1e6
        self.Nchannels = 373851
        self.G_LNA = 39 
        self.Lcable = -1
        self.antennaEfficiency = 0.75
        self.data = None
        self.Start_freq=0
        self.Stop_freq=0
        self.scaling_factor=5000           # number of points full data set
        
        self.zoom_data = None
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
        
        self.entry_data = self.makeform()
        self.newRWB_entry_data = self.new_res_BW()
        self.canvas = None
        self.figure = None
        self.fig_plot = None
        self.toolbar = None
        
        
#        self.CCFdata = self.get_CCF(AcqBW)
        self.GainCircuitData = self.readIQDatafileCSV(PathGainCircuit,FileGainCircuit)
        root.bind('<Return>', (lambda event,e=self.entry_data: self.fetch()))
        
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
                                       command=self.load_data,
                                       **DEFAULT_STYLE_2).pack(side=BOTTOM,fill=BOTH,expand=1,anchor=CENTER)
        
        # button to get new BW of the display resolution 
        self.newRBW_opts_frame = Frame(self.parent.newRBW_button_frame,pady=1)
        
        self.newRBW_opts_frame.pack(side=RIGHT,fill=BOTH,expand=1)
        self.show_newRBW_button=self._custom_button(
                self.newRBW_opts_frame,text = 'Replot', 
                command=(lambda e=self.newRWB_entry_data: self.zoom_act()),
                **DEFAULT_STYLE_2)
        
        self.fetch_opts_frame = Frame(self.parent.buttons_frame,pady=1)
        self.fetch_opts_frame.pack(side=TOP,fill=X,expand=1)
        self.button_fetch = self._custom_button(
                self.fetch_opts_frame,'Accept',
                command = (lambda e=self.entry_data: self.fetch()),
                **DEFAULT_STYLE_2)
        
        self.clear_opts_frame = Frame(self.parent.buttons_frame,pady=1)
        self.clear_opts_frame.pack(side=TOP,fill=X,expand=1)
        self.clear_button=self._custom_button(
                self.clear_opts_frame,text = 'Clear Plot', 
                command=(lambda e=self.entry_data: self.clear_data()),
                **DEFAULT_STYLE_2)
        
        # button to show plot
        self.plot_opts_frame = Frame(self.parent.buttons_frame,pady=1)
        self.plot_opts_frame.pack(side=TOP,fill=X,expand=1)
        self.show_plot_button=self._custom_button(
                self.plot_opts_frame,text = 'Show Plot', 
                command=(lambda e=self.entry_data: self.calibrateData(self.fig_plot,self.canvas)),
                **DEFAULT_STYLE_2)
        
        self.misc_opts_frame = Frame(self.parent.buttons_frame,pady=1)
        self.misc_opts_frame.pack(side=TOP,fill=X,expand=1)
        self.quit_button = self._custom_button(
        self.misc_opts_frame,"Quit",
        self.quit,**DEFAULT_STYLE_3)
        
             
    def plot(self):
        self.figure = Figure(figsize=(10,10))
        self.fig_plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent.top_frame_plot)
        self.canvas.get_tk_widget().pack(side=BOTTOM, fill=X, padx=5, pady=5)
        self.fig_plot = self.figure.add_subplot(111)
        self.toolbar = NavSelectToolbar(self.canvas,self.parent.top_frame_plot,self)
        self.toolbar.update()

    def zoom_act(self):
        self.zoom_Start_freq = float(self.newRWB_entry_data["Start Frequency (MHz): "].get())*1e6
        self.zoom_Stop_freq = float(self.newRWB_entry_data["Stop Frequency (MHz): "].get())*1e6
        self.zoom_max = float(self.newRWB_entry_data["Maximum amplitude: "].get())
        self.zoom_min = float(self.newRWB_entry_data["Minimum amplitude: "].get())
        self.zoom_trigger = True
        print("New displayed start frequency (MHz): %s"%self.zoom_Start_freq )
        print("New displayed stop frequency (MHz): %s"%self.zoom_Stop_freq)
        Bandwidth = self.Stop_freq - self.Start_freq
        print("True bandwidth: %s"%Bandwidth)
        zoom_Bandwidth = self.zoom_Stop_freq - self.zoom_Start_freq
        print("Zoom bandwidth: %s"%zoom_Bandwidth)
        self.clear_plot()
        max_nr_smp = int(5000/(40))
        zoom_nr_smp = int(zoom_Bandwidth/(40*1e6))
        print(self.zoom_data[0])
        if zoom_nr_smp >= int(max_nr_smp/2):
            self.scaling_factor = int((max_nr_smp/2)*10)
            print('Bandwidth/2: %f and scaling factor: %f' %(Bandwidth/2, self.scaling_factor))
        elif zoom_nr_smp < int(max_nr_smp/2) and zoom_nr_smp >= int(max_nr_smp/3):
            self.scaling_factor = int((max_nr_smp/3)*100)
            print('Bandwidth/3: %f and scaling factor: %f' %(Bandwidth/3,self.scaling_factor))
        elif zoom_nr_smp < int(max_nr_smp/3) and zoom_nr_smp >= int(max_nr_smp/4):
            self.scaling_factor = int((max_nr_smp/4)*1000)
            print('Bandwidth/4: %f and scaling factor: %f' %(Bandwidth/4,self.scaling_factor))
        elif zoom_nr_smp < int(max_nr_smp/4) and zoom_nr_smp >= int(max_nr_smp/5):
            self.scaling_factor = int((max_nr_smp/5)*10000)
            print('Bandwidth/5: %f and scaling factor: %f' %(Bandwidth/5,self.scaling_factor))
        elif zoom_nr_smp <= 4:
            self.scaling_factor = int(len(self.zoom_data[1])/5)
            print('40MHz and scaling factor: %f' %(self.scaling_factor))
            self.original = True
        self.plot_data(self.zoom_data[1])    
        
        
    def plot_data(self,data):
        reduced_data = self.read_reduce_Data(data) 
        if self.original == True:
            self.original = False
            data = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[1], color=color[0])
            self.fig_plot.set_ylim(self.zoom_min,self.zoom_max)
            self.fig_plot.set_xlim(self.Start_freq/1e6,self.Stop_freq/1e6)
            self.fig_plot.legend(['Original data'])
            self.canvas.draw()
        else:
            if self.zoom_trigger == True:
                self.fig_plot.set_ylim(self.zoom_min,self.zoom_max)
                self.fig_plot.set_xlim(self.zoom_Start_freq/1e6,self.zoom_Stop_freq/1e6)
                self.zoom_trigger = False 
            else:
                self.fig_plot.set_xlim(self.Start_freq/1e6,self.Stop_freq/1e6)
                self.fig_plot.set_ylim(np.min(reduced_data[2])-30,np.max(reduced_data[1])+30)
            
            max_plot, = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[1], color=color[0])
            min_plot, = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[2], color=color[1])
            mean_plot, = self.fig_plot.plot(reduced_data[0]/1e6,reduced_data[3], color=color[2])
            self.fig_plot.legend([max_plot,min_plot,mean_plot],['max', 'min', 'mean'])
            self.canvas.draw()
        
    def read_reduce_Data(self,spectrum):
        # read in the whole BW in one array

# set the display sample size depending on the display bandwidth and resolution         
        x = int(len(spectrum)/self.scaling_factor)
        if self.original == True:
            data = self.zoom_data
        else:
            spec_mean = np.array([])
            spec_min = np.array([])
            spec_max = np.array([])
            freq = np.array([])
            for i in range(self.scaling_factor): 
                
                temp_spec_max = np.max(spectrum[i*x:x*(i+1)])
                spec_max = np.append(spec_max, temp_spec_max)
                   
                temp_spec_mean = np.mean(spectrum[i*x:x*(i+1)])
                spec_mean = np.append(spec_mean, temp_spec_mean)
                    
                temp_spec_min = np.min(spectrum[i*x:x*(i+1)])
                spec_min = np.append(spec_min, temp_spec_min)
                    
            freq = np.linspace(self.Start_freq,self.Stop_freq,len(spec_max))
            temp = freq,spec_max,spec_min,spec_mean
            data = np.array(temp, dtype=np.float32)
        return data        
    
    def clear_plot(self):
        self.canvas.get_tk_widget().destroy()
        self.toolbar.destroy()
        self.canvas = None
        self.plot()
        
    def clear_data(self):
        answer = tkMessageBox.askyesno(title="Clear Plot and Data", message="Are you sure you want to clear the loaded data?")
        if (answer):
            self.zoom_data = None
            tkMessageBox.showwarning(title="Clear Plot and Data", message="Data has been deleted.")
            self.clear_plot()
            
    def new_res_BW(self):
        entries = {}
        for field in fields_zoom:
            row = Frame(self.parent.newRBW_button_frame)
            lab = Label(row, width=22, text=field, anchor='w')
            ent = Entry(row, width=22)
            row.pack(side=TOP, fill=X, padx=5, pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT, expand=NO, fill=X)
            entries[field] = ent
            print(ent)
        return entries
            
    def load_data(self):
        spec = np.array([])
        new_path = np.array([])
        path = self.directory_entry.get()
        path.replace("\\","/") + "/"
        new_path = path.split(' ')
#        Freqlist =cal.generate_CFlist(int(Start_freq),int(Stop_freq))
        center_freq_temp = []
        for i in range(len(new_path)): # i = 40 Mhz range
            CenterFrequency = re.findall("\d+", new_path[i])#re.search(r'\f+',new_path[i]).group()
       #     filename = self.Filename+str(i/1e6)+"MHz.npy"
            spectrum_temp = np.load(new_path[i])
       #     freq =  spectrum_temp[:,0].T*1e6
            spectrum =  spectrum_temp[:,1].T
            spec = np.append(spec, spectrum)
            center_freq_temp.append(float(CenterFrequency[0])*1e6)
            
#        Bandwidth = max(center_freq_temp)-min(center_freq_temp)
        self.Stop_freq = max(center_freq_temp) + self.Bandwidth/2
        self.Start_freq = min(center_freq_temp) - self.Bandwidth/2    
        print('Start freqeuncy (Hz): %f \nStop freqeuncy (Hz): %f'%(self.Start_freq, self.Stop_freq))
        freq = np.linspace(self.Start_freq,self.Stop_freq,len(spec))
        temp = freq,spec
        self.zoom_data = np.array(temp, dtype=np.float32) 
        data = np.array(spec, dtype=np.float32)                            # not reduced data
        self.plot_data(data)
        
    def load_zoom_data(self):
        # read in the whole BW in one array
        spec = np.array([])
        freq = np.array([])
        path = self.Path
        Stop_freq = self.CenterFrequency + self.Bandwidth/2
        Start_freq = self.CenterFrequency - self.Bandwidth/2
        #res = self.true_BW/self.Nsample
# set the display sample size depending on the display bandwidth and resolution 
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
            
    def fetch(self):          
        self.Path= self.entry_data['Path'].get()
        self.Filename = self.entry_data['Filename'].get()
        self.Start_freq = self.entry_data['Start Frequency (Hz)'].get()
        self.Stop_freq = self.entry_data['Stop Frequency (Hz)'].get()
        self.G_LNA = self.entry_data['LNA gain (dB)'].get() 
        self.Lcable = self.entry_data['Cable losses (dB)'].get()
        self.antennaEfficiency = self.entry_data['Antenna efficiency'].get()
        self.get_x_factor = self.entry_data['Set x axes scaling factor'].get()
        #self.Path, self.Filename, self.Start_freq, self.Stop_freq, self.G_LNA, self.Lcable, self.antennaEfficiency)
        for field in fields: 
            print(field+': %s \n' % (self.entry_data[field].get()))
        #    print(field': %s \n' % (self.Path, self.Filename, self.Start_freq, self.Stop_freq, self.G_LNA, self.Lcable, self.antennaEfficiency)) 
           
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

    def makeform(self):
       entries = {}
       for field in fields:
           row = Frame(self.parent.input_frame)
           lab = Label(row, width=22, text=field+": ", anchor='w')
           ent = Entry(row, width=22)
           # ent.insert(0,"0")
           row.pack(side=TOP, fill=X, padx=5, pady=5)
           lab.pack(side=LEFT)
           ent.pack(side=RIGHT, expand=NO, fill=X)
           entries[field] = ent 
           print(field+': %s \n' % (entries[field].get()))
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


    
    def readSpectralData(self,path,filename):
        arraydata = np.load(path + filename)
        return arraydata
    
    def calCCF(self,spectrum, CCF, r, Lcable, G_LNA, antennaEfficiency): # returns in [dBuV/m]
         # spectrum numpy array
         #get the 1MHz average of the spectrum
         temp = -G_LNA - Lcable - (10.0 * np.log10(antennaEfficiency)) - CCF + (10.0 * np.log10(Z0 / (4.0 * np.pi * (r*r)))) + 90.0
         
         return spectrum[0], spectrum[1]+temp
     
    def calibrateData(self,fig_plot,canvas):
       tstart = time.time()
       temp_maxSpec = []
       temp_minSpec = []
       if len(self.entry_data) == 0:
           print('Invalid directory defined')
       else:
#           G_LNA = int(self.G_LNA) 
 #          G_LNA = self.cal_GainCircuit()
 #          Lcable = int(self.Lcable)
#           antennaEfficiency = np.float32(self.antennaEfficiency) 
           spectrum = self.read_reduce_Data()
           #spec = self.reduced_Data(spectrum_whole_bw,40)
           #smooth_spectrum_whole_bw = smooth(spectrum_whole_bw[1])
           CCF = self.get_CCF()
           
           #Spec = self.calCCF(spec, CCF, r, int(Lcable), G_LNA, float(antennaEfficiency))
           #temp_maxSpec.append(max(Spec[1]))
           #temp_minSpec.append(min(Spec[1]))
           cal_spec = fig_plot.plot(spectrum[0]/1e6,spectrum[1], color=color[0])#,label = 'Calibarted data')
           #raw_spec, = self.fig_plot.plot(spec[0]/1e6,spec[1], color=color[1],label='Raw data')
           #self.fig_plot.legend()#([cal_spec,raw_spec],['Calibarted data', 'Raw data'])
#           maxSpec = max(temp_maxSpec)
#           minSpec = min(temp_minSpec)
#           self.fig_plot.axis([int(self.Start_freq),int(self.Stop_freq), minSpec-self.x_factor, maxSpec+self.x_factor])
           fig_plot.set_ylabel("Electrical Field Strength [dBuV/m]")#('Power [dBm]')
           fig_plot.set_xlabel("Frequency (MHz) (resolution %.3f kHz)"%1)
#           print('Path: %s \nFilename: %s \nStart Frequency (MHz): %s \nStop Frequency (MHz): %s \nLNA gain (dB): %s\nCable losses (dB): %s \nAntenna Efficiency: %s \n' % (self.Path, self.Filename, self.Start_freq, self.Stop_freq, self.G_LNA, self.Lcable, self.antennaEfficiency)) 
           canvas.draw()
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