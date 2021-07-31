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

from solvers.line_search import NewtonMethod
def sysIDDemo(controls,magnets,positions,linearVelocity,rotationalVelocity):

    optimizerParams = {'alpha':1,'ro':0.9,'c1':1e-9,'FP_PRECISION': 1e-5,'min_alpha': 1e-20}
    prob = radiusSysid(np.diag((2e5,2e5,2e5,1,1,1))) # built with M(0.04) Q M(0.04)
    optimizer = NewtonMethod(problem=prob,**optimizerParams)
    prob.updateVals(controls,magnets,positions,linearVelocity,rotationalVelocity)
    solution, cost, converged, iterates = optimizer.optimize(prob.initial_solution,max_iterations=25)
    prob.visualize_optimization(iterates)
    print(solution)



class radiusSysid(Problem): 
    def __init__(self,weights): 
        super().__init__()
        self.boundConstraints[0] = (1e-3,0.1)
        self.singleModel = sphereModel(1)
        self.initial_solution = np.ones((1,1)) * 0.05
        
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
        
        error = (self.accelerations - torch.linalg.inv(M)@ wrenches.T)*1e6
        cost = torch.sum(error.T @ self.weights @ error)

        return cost
    def visualize_optimization(self,iterates): 
        
        func = np.abs
        possible = np.linspace(1e-3,0.1,5000).reshape((-1,1,1))
        possibleCosts = [self.cost(decision) for decision in possible]
        plt.figure()
        plt.plot(possible[:,0,0],func(possibleCosts),'r')
        foundCosts = [self.cost(iterate) for iterate in iterates]
        plt.plot(iterates[:,0,0],func(foundCosts),'b')
        print('best brute force ',possible[np.argmin(possibleCosts)])
        

