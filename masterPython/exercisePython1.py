# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:35:37 2019

@author: geomarr
"""
import numpy as np
# classes create objects
class Validate_Str_Parentheses:
    
    def check(self, str_sym):
        for num1, val1 in enumerate(str_sym):
            if val1 == '(':
                if str_sym.find(')')>0:
                    return True
                else:
                    return False
            if val1 == '[':
                    if str_sym.find(']')>0:
                        print('true')
                    else:
                        print('false')
                
if __name__ == '__main__':
    test = Validate_Str_Parentheses()
    print(Validate_Str_Parentheses().check("(){}[]"))
    print(Validate_Str_Parentheses().check("()[{)}"))
    print(Validate_Str_Parentheses().check("()"))
    
 
        
    