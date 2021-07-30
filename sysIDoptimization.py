#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 20:30:42 2021

@author: tabor
"""

import numpy as np
import time
import torch
from models import modelCallRotatedWrench,sphereModel
from discreteTimeSimulator import buildMassMatrix,sphereProperties
from modelOptimization import buildQ
from problems.problem import Problem 
import matplotlib.pyplot as plt 


class radiusSysid(Problem): 
    def __init__(self,weights): 
        super().__init__()
        self.boundConstraints[0] = (1e-5,0.1)
        self.singleModel = sphereModel(1)
        self.initial_solution = np.ones((1,1)) * 1e-3
        
        self.weights = torch.tensor(weights)
    def updateVals(self,controls,magnets,positions,linearVelocity,rotationalVelocity):
        
        #remove final control and position as we dont know its acceleration
        npControl = np.array(controls[:-1])
        magNums = npControl[:,0].astype(int).tolist()
        chosenMagnetPose = np.array(magnets)[magNums]-positions[:-1]
    
    
        dipoleStrength = npControl[:,1]
        angles = npControl[:,2:].T
        rhos = -chosenMagnetPose.T
        dists=np.linalg.norm(chosenMagnetPose,axis=1)
        zero = np.array([0]*npControl.shape[0])

        self.dipoleStrength = torch.tensor(dipoleStrength)
        self.angles = torch.tensor(angles)
        self.rhos = torch.tensor(rhos)
        self.dists = torch.tensor(dists)
        
        self.zero = torch.tensor(zero)

        self.n = npControl.shape[0]
        
        npA = np.hstack((linearVelocity[1:,:]-linearVelocity[:-1,:],rotationalVelocity[1:,:]-rotationalVelocity[:-1,:]))        
        self.accelerations = torch.tensor(npA.T)
        
        self.materialDensity = torch.tensor(8940.0)
        
    def tensor_cost(self, x):
        radii = x

        modelVals = self.singleModel.modelCall(self.dists,1,radii=radii)
        modelVals = [modelVal.squeeze() for modelVal in modelVals]

        wrenches = modelCallRotatedWrench(self.dipoleStrength,self.angles,modelVals,self.rhos,library=torch,zero = self.zero).squeeze()
        properties = sphereProperties(self.materialDensity,x*2,torch)
        M = buildMassMatrix(properties,torch,False)

        Minv = torch.linalg.inv(M)

        expectedAccel = Minv @ torch.swapaxes(wrenches,0,1)

        error = (self.accelerations - expectedAccel)*1e6
        cost = torch.sum(error.T @ self.weights @ error)

        return cost
    def visualize_optimization(self,iterates): 
        
        func = np.log
        possible = np.linspace(1e-10,0.1,500).reshape((-1,1,1))
        possibleCosts = [self.cost(decision) for decision in possible]
        plt.figure()
        plt.plot(possible[:,0,0],func(possibleCosts),'r')
        foundCosts = [self.cost(iterate) for iterate in iterates]
        plt.plot(iterates[:,0,0],func(foundCosts),'b')

        

