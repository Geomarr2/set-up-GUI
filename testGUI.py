# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 10:47:07 2018

@author: User
"""

from tkinter import *
from math import *

def evaluate(event):
    res.configure(text = "Ergebnis: " + str(eval(entry.get())))
    

if __name__ == '__main__':
    w = Tk()
    Label(w, text="Your Expression:").pack()
    entry = Entry(w)
    entry.bind("<Return>", evaluate)
    entry.pack()
    res = Label(w)
    res.pack()
    Button(w, text='Quit', command=w.quit).pack()
    
    w.mainloop()