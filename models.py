#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 17:23:35 2021

@author: tabor
"""
import numpy as np
import torch

def getOmniFrame(pos,use_tf=False):
    Z = np.array([0, 0, 1])

    dist = np.linalg.norm(pos)

    omni_Z = -1*(pos/dist);
    
    omni_Y = np.cross(omni_Z,Z)
    omni_Y = omni_Y/np.sqrt(np.dot(omni_Y,omni_Y)) #unit vectorise
    
    omni_X = np.cross(omni_Y,omni_Z);
    # Combine to build rotation matrix
    omniFrame = np.transpose(np.array([omni_X, omni_Y, omni_Z]));
    return omniFrame


class cylinderModel():
    def __init__(self,diameter):
        self.time = 0.0
        sigma_copper = 5.8E+07;
        sigma_al = 3.77E+07;
        self.sigma = sigma_copper;
        self.do = diameter;
        self.radii = diameter/2;
        self.freq = 15
        self.mu = 4*np.pi*1e-7;  
        
        self.feaConstants = {}
        self.feaConstants['fzz'] = [430,2.95,-0.101,-9.26,7]
        self.feaConstants['tzz'] = [6840,3.00,-0.0986,-13.2,6]
        self.feaConstants['frhorho'] = [266,2.60,-0.101,-7.65,7]
        self.feaConstants['frhophi'] = [6040,3.45,-0.102,-14.3,7]
        self.feaConstants['trhoz'] = [8000,3.40,-0.0928,-15.0,6]

    def modelEquation(self,constants,distance,dipoleStrength,p0RPower):
        #P0 has r^4 for forces and r^3 for torques
        P1 = self.sigma*self.mu*self.freq*self.radii*self.radii
        P2 = distance/self.radii
        
        numeratorBase = constants[0]*P1
        numeratorPower = constants[1]*(P1**constants[2])
        numerator = (numeratorBase**numeratorPower) * (10**constants[3])
        
        denominator = P2**constants[4]
        P0 = numerator/denominator
        
        c=  self.mu*(dipoleStrength**2)*(self.radii**-P0RPower)

        wrench = P0*c
        
        return wrench
    def modelCall(self,dist,dipoleStrength):
        constants = self.feaConstants
        
        fzz = self.modelEquation(self.constants['fzz'],dist,dipoleStrength,4)
        tzz = self.modelEquation(self.constants['tzz'],dist,dipoleStrength,3)
        
        frhorho = self.modelEquation(self.constants['frhorho'],dist,dipoleStrength,4)
        frhophi = self.modelEquation(self.constants['frhophi'],dist,dipoleStrength,4)
        trhoz = self.modelEquation(self.constants['trhoz'],dist,dipoleStrength,3)
        
        pointingModel = np.array([0,0,fzz,0,0,tzz]).reshape((6,1))
        negPointingModel = np.array([0,0,fzz,0,0,-tzz]).reshape((6,1))
        
        rhoModel = np.array([0,-frhophi,frhorho,-trhoz,0,0]).reshape((6,1))
        return pointingModel,negPointingModel,rhoModel



