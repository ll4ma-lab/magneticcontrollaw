# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:57:35 2019

@author: tabor
"""

import numpy as np
class exponentialSimulator:
    def __init__(self,state,timestep,object_properties,unitConversionFactor = 1.0,taylorDepth = 3):
        (mass,MOI) = object_properties
        self.state = np.copy(state)
        self.t = timestep
        self.tayDepth = taylorDepth
        self.mass = mass
        self.MOI = MOI
        self.unitFactor = unitConversionFactor
        self.matrixExponential(timestep)
    def getScaledSimulator(self,scale):
        newSim = exponentialSimulator(self.state*scale,self.t,(self.mass,self.MOI),self.unitFactor*scale,self.tayDepth)
        return newSim
    def matrixExponential(self,timestep):
        stateDims = len(self.state)
        halfStateDims = stateDims/2
        quarterDims = halfStateDims/2
        Ac = np.zeros((stateDims,stateDims))
        Ac[:halfStateDims,halfStateDims:] = np.eye(halfStateDims) 
        
        Bc = np.zeros((stateDims,halfStateDims))
        Bc[halfStateDims:,:halfStateDims] = np.eye(halfStateDims)
        Bc[halfStateDims:halfStateDims + quarterDims,:quarterDims] *= (self.unitFactor/self.mass)
        Bc[halfStateDims + quarterDims:,quarterDims:] = self.unitFactor*np.linalg.inv(self.MOI)

        #Bc[halfStateDims:halfStateDims+quarterDims,:quarterDims] = np.eye(quarterDims) * 1.0/self.mass
        
        fullContinious = np.zeros((stateDims+halfStateDims,stateDims+halfStateDims))        
        fullContinious[:stateDims,:stateDims] = Ac
        fullContinious[:stateDims,stateDims:] = Bc
        fullContinious[stateDims:,stateDims:] = np.eye(halfStateDims) 
        
        fullDiscrete = np.zeros((stateDims+halfStateDims,stateDims+halfStateDims))        
        for i in range(self.tayDepth):
            nextD = np.linalg.matrix_power(fullContinious,i) * np.power(timestep,i)
            fullDiscrete += nextD
        
        self.Ad =  fullDiscrete[:stateDims,:stateDims]
        self.Bd =  fullDiscrete[:stateDims,stateDims:]
        
       # self.Bd [halfStateDims:,:halfStateDims] *= 1.0/self.mass
    def simulate(self,control):
        self.state = self.forwardPass(self.state,control)
        return self.state
    def forwardPass(self,pose,control):
        newPose = np.dot(self.Ad, pose) + np.dot(self.Bd,control)
        return newPose
    def setState(self,state):
        self.state = np.copy(state)
def sphereProperties(density,diameter):
    radius = diameter / 2.0
    volume = 4.0/3 * np.pi*np.power(radius,3)
    mass = volume * density
    I = np.eye(3)*(2.0/5) *mass * np.power(radius,2)
    return mass, I
