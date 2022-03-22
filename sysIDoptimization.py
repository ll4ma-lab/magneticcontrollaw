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
from discreteTimeSimulator import buildMassMatrix,sphereProperties,boatProperties
from modelOptimization import buildQ
from ll4ma_opt.problems import Problem 
import matplotlib.pyplot as plt 

from ll4ma_opt.solvers.line_search import NewtonMethod
def sysIDDemo(controls,magnets,positions,linearVelocity,rotationalVelocity):

    optimizerParams = {'alpha':1,'ro':0.9,'c1':1e-9,'FP_PRECISION': 1e-5,'min_alpha': 1e-20}
    prob = radiusSysid(np.diag((2e5,2e5,2e5,1,1,1))) # built with M(0.04) Q M(0.04)
    optimizer = NewtonMethod(problem=prob,**optimizerParams)
    prob.updateVals(controls,magnets,positions,linearVelocity,rotationalVelocity)
    result = optimizer.optimize(prob.initial_solution,max_iterations=25)
    prob.visualize_optimization(result.iterates)
    print(result.solution)



class radiusSysid(Problem): 
    def __init__(self,weights,physical = False): 
        super().__init__()
        self.singleModel = sphereModel(1)
        self.initial_solution = np.ones((3,1)) * 0.001
        self.initial_solution[1] = 5.8E+07
        self.initial_solution[2] = 0.00067

        self.min_bounds[0] = 1e-2
        self.max_bounds[0] = 1e-1

        self.min_bounds[1] = 2.5e7
        self.max_bounds[1] = 6e7
        
        self.min_bounds[2] = self.initial_solution[2]-1e-10
        self.max_bounds[2] = self.initial_solution[2]+1e-10

        self.weights = torch.tensor(weights)
        self.physical = physical
        if(physical):
            self.boat_mass, boat_MOI = boatProperties(0.059,0.075,0.07,0.031)
            self.boat_MOI = torch.tensor(boat_MOI)
        
    def updateVals(self,controls,magnets,positions,linearVelocity,rotationalVelocity):
        #remove first control and position as we dont know its acceleration
        npControl = np.array(controls[1:]).reshape((-1,4))
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
        sigma = x[1:2,0]
        boat_MOIzz = x[2:3,0]
        
        modelVals = self.singleModel.modelCall(dists,1,radii=radii,sigma=sigma)
        modelVals = [modelVal.squeeze() for modelVal in modelVals]

        wrenches = modelCallRotatedWrench(dipoleStrengths,angles,modelVals,rhos,library=torch,zero = zero).squeeze()
    
        mass = sphereProperties(self.materialDensity,radii,torch)
        objectProperties = {'mass':mass[0],'MOI':mass[1],'radius':radii}
        if self.physical:
            objectProperties['mass'] += self.boat_mass
            objectProperties['MOI'] = objectProperties['MOI'] + self.boat_MOI
            
        M = buildMassMatrix(objectProperties,torch,False)
        
        return torch.linalg.inv(M)@ wrenches.T
    
    def size(self):
        return 3
    
    def tensor_cost(self, x):

        modeledAccelerations = self.forwardAcceleration(x,self.dists,self.dipoleStrength,self.angles,self.rhos,self.zero)
        error = (self.accelerations - modeledAccelerations)*1e6 
        
        
        left = (error.T @ self.weights).unsqueeze(1)
        right = error.T.unsqueeze(2)
        cost = torch.sum(torch.bmm(left,right))#/ self.rhos.shape[0] #-  5e14* (x[0:1])**2

        return cost
    def visualize_optimization(self,iterates,func = np.abs): 
        possible_r = np.linspace(self.min_bounds[0],self.max_bounds[0],50)
        possible_sigma = np.linspace(self.min_bounds[1],self.max_bounds[1],50)
        rs,sigmas = np.meshgrid(possible_r,possible_sigma)
        rs = rs.flatten()
        sigmas = sigmas.flatten()
        possibleCosts = [self.cost(np.stack((rs[i],sigmas[i],1)).reshape((3,1))) for i in range(len(rs))]
        clip_value = 2*np.median(possibleCosts)
        possibleCosts = np.array(possibleCosts)
        possibleCosts[possibleCosts>clip_value] = clip_value
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(rs,sigmas,func(possibleCosts),'r')
        foundCosts = [self.cost(iterate) for iterate in iterates[1:]]
        ax.plot(iterates[1:,0,0],iterates[1:,1,0],func(foundCosts),'r',linewidth=10)
        print('best brute force ',rs[np.argmin(possibleCosts)],sigmas[np.argmin(possibleCosts)],np.min(possibleCosts))
        return rs,sigmas,possibleCosts

