# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:41:12 2019

@author: tabor
"""
from oct2py import octave
import numpy as np
class matlabModel:
    def __init__(self,magnets):
        self.magnets = magnets
        self.dims = len(magnets)*3
        pass
    def getAB(self,pose):
        sigma_copper = 5.8E+07;
        sigma_al = 3.77E+07;
        sigma = sigma_copper;
        do = 0.05;
        allFT,A,B = octave.calcEddyInput(self.magnets,sigma,do,nout=3)
        A=A[:,:self.dims] * 10**6
        B=B[:,:self.dims] * 10**6
        return A,B
    def forwardPass(self,pose,control):
        A,B = self.getAB(pose)
        #print(A.shape)
        #print(B.shape)
        #print(control.shape)
        FT = np.add(np.matmul(A,control),np.matmul(B,np.abs(control)))
        return FT
