"""
author: G-man
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

class windowToDisplayInfo:
    def __init__(self, root,headerFile):     
        self.root = root
        self.root.title("Acquisition information")
        self.text_frame = Frame(self.root)
        self.text_frame.pack(side=TOP)
        self.text = Text(self.text_frame)
        scroll = Scrollbar(root, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.tag_configure('bold_italics', font=('Arial', 12, 'bold', 'italic'))
        self.text.tag_configure('big', font=('Verdana', 20, 'bold'))
        self.text.tag_configure('color', foreground='#476042', 
						font=('Tempus Sans ITC', 12, 'bold'))

        
        self.printHeaderInfo(headerFile)
        
        
    def printHeaderInfo(self,headerFile):
#Load in all the header file get the info and the BW from max and min center freq
        print(headerFile)
        #file = open(headerFile[0], "r")
        headerInfo = file.readlines()
        
        for x in headerInfo:
            print(x)
        file.close()