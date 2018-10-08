# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:27:02 2018

@author: User
"""
import numpy as np
import ctypes as c

def hello(to=__name__):
    return "hello, %s" % to

if __name__ == "__main__":
    print(hello("world"))