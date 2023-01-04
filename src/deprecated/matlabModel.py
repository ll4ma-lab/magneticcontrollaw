# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:41:12 2019

@author: tabor
"""
from oct2py import octave
import numpy as np
import time

class matlabModel:
    def __init__(self,magnets,fast=True):
        self.magnets = magnets
        self.dims = len(magnets)*3
        self.A = None
        self.B = None
        self.fast = fast
        self.time = 0.0
    def getAB(self,pose):
        if(self.fast and self.A is not None):
            return self.A,self.B
        position = pose[:3]
        newMagnets = (np.array(self.magnets) - np.transpose(position)).tolist()
        sigma_copper = 5.8E+07;
        sigma_al = 3.77E+07;
        sigma = sigma_copper;
        do = 0.05;
        start = time.time()
        allFT,A,B = octave.calcEddyInput(newMagnets,sigma,do,15,nout=3)
        runTime = time.time() - start
        self.time = self.time + runTime
        A=A[:,:self.dims] #* 10**6
        B=B[:,:self.dims] #* 10**6
        self.A = A
        self.B = B
        return A,B
    def forwardPass(self,pose,control):
        A,B = self.getAB(pose)
        #print(A.shape)
        #print(B.shape)
        #print(control.shape)
        FT = np.add(np.dot(A,control),np.dot(B,np.abs(control)))
        return FT
