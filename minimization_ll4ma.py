# -*- coding: utf-8 -*- 
""" 
Created on Tue May 11 14:06:39 2021 
 
@author: Griffin Tabor 
""" 
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import numpy as np 


import matplotlib.pyplot as plt 
import time 
from PDController import PDController
from torch.autograd import Variable

from environments import PlanarProblem,SpaceProblem
import SplineProfile
from discreteTimeSimulator import discreteTimeSimulator,sphereProperties

from models import multiMagModel,multiMagSolver


env = SpaceProblem(4)
controller = PDController(env.startState,0.005,0.00001,0.005,0.00001,0.0,0.000)
velocityProfile = SplineProfile.SplineProfile(env,np.sum(env.times),env.times)


diameter = 0.04
objectProperties = sphereProperties(8940,diameter)

if __name__ == '__main__':

    position = np.zeros((3,1))
    newModel = multiMagModel(env.magnets,0.04)
    solver = multiMagSolver(newModel)
    #control = [0,2,np.random.random()*40,np.random.random()*360-180]
    control = [0, 2, 48.36714973571669, 172.5982599543529]
    desired_wrench = newModel.forwardPass(position,control)
    
    desired_wrench = (np.random.random((6,1))-0.5)*1e-4    
    
    
    
    s = time.time()
    
    solver.solve(desired_wrench, position)
    
    print('elapsed time ',time.time()-s)
    for i in range(10):
        s = time.time()
        desired_wrench = (np.random.random((6,1))-0.5)*1e-4    
        results = solver.solve(desired_wrench, position)
        
        
        print('elapsed time ',time.time()-s)

    