#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:47:32 2021

@author: tabor
"""

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import numpy as np
from PDController import PDController
from discreteTimeSimulator import discreteTimeSimulator,sphereProperties,simulateExperiment
from environments import PlanarProblem,SpaceProblem
from se3Utils import buildTF
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

import time
import visualization
import DummyProfile
import SplineProfile
import TrapProfile
from tqdm import tqdm

useSphere = False

from models import multiMagModel
from modelOptimization import multiMagSolver

if __name__ == '__main__':

    
    env = SpaceProblem(6)
    env.magnets = env.magnets[:-3]
    env.trajectoryDimensions = 'xyz'
    
    simTime = np.sum(env.times)
    dt = 1
    numSteps = int(simTime/dt)
    diameter = 0.04
    objectProperties = sphereProperties(8940,diameter)
    
    model = multiMagModel(env.magnets,diameter,useSphere)
    solverModel = multiMagModel(env.magnets,diameter,useSphere)
    solver = multiMagSolver(solverModel,env.trajectoryDimensions,useSphere,10)

    modifier = 10
    controller = PDController(env.startState,0.005*modifier,0.000002*modifier,0.005*modifier,0.000002*modifier,0.0,0.000)
    
    
    sampledVals = np.linspace(-3e-2,3e-2,16)
    x,y,z = np.meshgrid(sampledVals, sampledVals, sampledVals, indexing='ij')
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    samplePoints = np.stack((x,y,z)).T

    
    posErrorsPercent = []
    velErrors = []
    s = time.time()
    for i in tqdm(range(samplePoints.shape[0])):
        env.desiredState[:3,0] = samplePoints[i,:]
        simulator = discreteTimeSimulator(env.startState,dt,objectProperties) #density, diamter
        velocityProfile = SplineProfile.SplineProfile(env,simTime,env.times)
        states,wrenches,desiredWrenchs,desiredStates,controls,minimizationCosts = simulateExperiment(simulator,controller,env,velocityProfile,model,solver,numSteps=numSteps,cheat = False,dt=dt,debugPrints=False)
        
        velError = np.linalg.norm(np.dot(states[-1][0][:3,:3],states[-1][2][:,0]))
        velErrors.append(velError)
    

        desiredFinalTF = buildTF(desiredStates[-1][:3,0],desiredStates[-1][3:6,0])
        remainingError = controller.computeError(desiredFinalTF,states[-1][0]) 
        posError = np.linalg.norm(remainingError[:3])
        startingError = controller.computeError(desiredFinalTF,np.eye(4)) 
        startingPosError = np.linalg.norm(startingError[:3])
        posErrorsPercent.append(posError/startingPosError)
    
    
    threshold = 0.1
    colors = [(1,0,0,1-error/threshold if error<threshold else 0) for error in posErrorsPercent]
    print(time.time()-s)
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.scatter(samplePoints[:,0],samplePoints[:,1],samplePoints[:,2],c=colors)
    ax.set_xlim(*env.visualizationBoundaryY)
    ax.set_ylim(*env.visualizationBoundaryY)
    ax.set_zlim(*env.visualizationBoundaryY)
    visualization.drawCubes(env.magnets,[],ax,env.radius)
    
    
    name = 'sphere_' if useSphere else 'cylinder_'
    import dataloader
    import dataloader
    from datetime import datetime
    dataloader.save("../data/" + name +"_mags_" + str(len(env.magnets)) + '_samples_'+str(samplePoints.shape[0]) +  str(datetime.now()).replace(":","."),globals(),dir())