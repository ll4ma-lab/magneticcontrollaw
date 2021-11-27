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
from ll4ma_opt.problems import Problem 
import matplotlib.pyplot as plt 

from ll4ma_opt.solvers.line_search import NewtonMethod
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
        
        #remove first control and position as we dont know its acceleration
        npControl = np.array(controls[1:])
        magNums = npControl[:,0].astype(int).tolist()
        chosenMagnetPose = np.array(magnets)[magNums]-positions[1:]
    
    
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
    def forwardAcceleration(self, x,dists,dipoleStrengths,angles,rhos,zero):
        radii = x[0:1]

        modelVals = self.singleModel.modelCall(dists,1,radii=radii)
        modelVals = [modelVal.squeeze() for modelVal in modelVals]

        wrenches = modelCallRotatedWrench(dipoleStrengths,angles,modelVals,rhos,library=torch,zero = zero).squeeze()
    
        mass = sphereProperties(self.materialDensity,radii,torch)
        objectProperties = {'mass':mass[0],'MOI':mass[1],'radius':radii}
        
        M = buildMassMatrix(objectProperties,torch,False)
        
        return torch.linalg.inv(M)@ wrenches.T
    def tensor_cost(self, x):

        modeledAccelerations = self.forwardAcceleration(x,self.dists,self.dipoleStrength,self.angles,self.rhos,self.zero)
        error = (self.accelerations - modeledAccelerations)*1e6
        
        
        left = (error.T @ self.weights).unsqueeze(1)
        right = error.T.unsqueeze(2)
        cost = torch.sum(torch.bmm(left,right))

        return cost
    def visualize_optimization(self,iterates,func = np.abs): 
        possible = np.linspace(1e-3,0.1,5000).reshape((-1,1,1))
        possibleCosts = [self.cost(decision) for decision in possible]
        plt.figure()
        plt.plot(possible[:,0,0],func(possibleCosts),'r')
        foundCosts = [self.cost(iterate) for iterate in iterates]
        plt.plot(iterates[:,0,0],func(foundCosts),'b')
        print('best brute force ',possible[np.argmin(possibleCosts)])
        return possible,possibleCosts

