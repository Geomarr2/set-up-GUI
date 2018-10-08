# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:41:08 2018

@author: User
"""

import tkinter
from tkinter import *

class simpleapp_tk(tkinter.Tk):
    def __init__(self,parent):                                                  # So each GUI element has a parent (its container, usually).
                                                                                # That's why both constructor have a parent parameter.
                                                                                # Keeping track of parents is usefull when we have to show/hide a group of widgets, 
                                                                                # repaint them on screen or simply destroy them when application exits.
        tkinter.Tk.__init__(self,parent)
        self.parent = parent                                                    # Keep track of parent
        self.initialize()

    def initialize(self):
        self.grid()
    
if __name__ == "__main__":
    app = simpleapp_tk(None)
    app.title('G-man')
    app.mainloop()
    