# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:51:15 2018

@author: geomarr
"""
try:
    # Python2
    from Tkinter import * 
except ImportError:
    # Python3
    from tkinter import *
    
import ctypes as c
import os
import numpy as np
import math as math
import matplotlib.pyplot as plt
import RFIHeaderFile
import DataLoad
from tkinter import messagebox as tkMessageBox
from tkinter import filedialog as tkFileDialog

BESTPROF_DTYPE  = [
    ("Text files","*.txt"),("all files","*.*")
    ]
#-----------Misc.-------------#
DEFAULT_DIRECTORY = "C:/Users/geomarr/Documents/GitHub/set-up-GUI/RFIDummyDataset2/"#os.getcwd()

#----------Style options---------#
DEFAULT_PALETTE = {"foreground":"blue","background":"lightblue"}
DEFAULT_STYLE_1 = {"foreground":"black","background":"lightblue"}
DEFAULT_STYLE_2 = {"foreground":"gray90","background":"darkgreen"}
DEFAULT_STYLE_3 = {"foreground":"gray90","background":"darkred"}

class readDataGUI:
    def __init__(self, root):     
        self.root = root
        self.x_factor = 20
        self.root.title("Load data")
        
        self.top_frame = Frame(self.root)
        self.top_frame.pack(side=RIGHT)
        self.dir_button_frame = Frame(self.top_frame)
        self.dir_button_frame.pack(side=BOTTOM)
        
        self.input_frame = Frame(self.top_frame)
        self.input_frame.pack(side=TOP)
        self.buttons_frame = Frame(self.input_frame)
        self.buttons_frame.pack(side=RIGHT)
        
        
        self.top_frame = Frame(self.dir_button_frame)
        self.top_frame.pack(side=TOP,anchor=W)
        
        self.bottom_frame = Frame(self.dir_button_frame)
        self.bottom_frame.pack(side=BOTTOM,fill=BOTH,expand=1)
        RFIHeaderFile.HeaderInformation.__init__(self)
        DataLoad.DataInfo.__init__(self)
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
                                       **DEFAULT_STYLE_2).pack(side=BOTTOM,fill=BOTH,expand=1,anchor=CENTER)
        
        self.plot_opts_frame = Frame(self.buttons_frame,pady=1)
        self.plot_opts_frame.pack(side=TOP,fill=X,expand=1)
        self.show_plot_button=self._custom_button(
                self.plot_opts_frame,text = 'Get Print Header File', 
                command=(lambda self=self: self.printHeaderInfo()),
                **DEFAULT_STYLE_2)
        
        self.plot_opts_frame = Frame(self.buttons_frame,pady=1)
        self.plot_opts_frame.pack(side=TOP,fill=X,expand=1)
        self.show_plot_button=self._custom_button(
                self.plot_opts_frame,text = 'Get BW', 
                command=(lambda self=self: self.findCenterFrequency()),
                **DEFAULT_STYLE_2)
        
        self.plot_opts_frame = Frame(self.buttons_frame,pady=1)
        self.plot_opts_frame.pack(side=TOP,fill=X,expand=1)
        self.show_plot_button=self._custom_button(
                self.plot_opts_frame,text = 'Load Data', 
                command=(lambda self=self: self.loadDataFromFile()),
                **DEFAULT_STYLE_2)
        

        
    def launch_dir_finder(self):
        #directory = tkFileDialog.askdirectory()
        files = tkFileDialog.askopenfilenames()
        self.directory_entry.delete(0,END)
        self.directory_entry.insert(0,files)
        
    def _custom_button(self,root,text,command,**kwargs):
        button = Button(root, text=text,
            command=command,padx=20, pady=20,height=1, width=10,**kwargs)
        button.pack(side=TOP,fill=BOTH)
        return button    
    
    def printHeaderInfo(self):
#Load in all the header file get the info and the BW from max and min center freq
        file = open(self.headerFile[0], "r")
        self.headerInfo = file.readlines()
        for x in self.headerInfo:
            print(x)
        file.close()
        
    def loadDataFromFile(self):
        spec = np.array([])
        for i in range(len(self.dataFile)):
            temp = np.load(self.dataFile[i])
            spec = np.append(spec, temp)
        freq = np.linspace(self.Start_freq,self.Stop_freq,len(spec))
        temp = freq,spec
        self.DATA = np.array(temp, dtype=np.float32)
        print(self.DATA)
        
    def dump_filenames(self):
#Load in vereything and seperate header and data file
        path = self.directory_entry.get()
        path.replace("\\","/") + "/"
        new_path = path.split(' ')
        for i in range(len(new_path)): 
            if new_path[i].endswith(".rfi"):
                self.headerFile.append(new_path[i])
            elif new_path[i].endswith(".npy"):
                self.dataFile.append(new_path[i])
        self.printHeaderInfo()
        self.findCenterFrequency()
        self.loadDataFromFile()
        
        
    def findCenterFrequency(self):
#Load in all the header file get the info and the BW from max and min center freq
        center_freq_temp = []
        for i in range(len(self.headerFile)): # i = 40 Mhz range
            temp_path = self.headerFile[i].split('_')
            name_file = temp_path[0]
            cfreq = temp_path[1]
            scanID_file = temp_path[2]
            center_freq_temp.append(float(cfreq))
            
        self.Stop_freq = max(center_freq_temp) + self.Bandwidth/2
        self.Start_freq = min(center_freq_temp) - self.Bandwidth/2    
        print('Start freqeuncy (Hz): %f \nStop freqeuncy (Hz): %f'%(self.Start_freq, self.Stop_freq))
        
if __name__ == '__main__':
   root = Tk()
   root.tk_setPalette(**DEFAULT_PALETTE)
   start = readDataGUI(root)

   
   root.mainloop() 