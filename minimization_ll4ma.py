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
    solver = multiMagSolver(newModel,'all')


    desired_wrench = (np.random.random((6,1))-0.5)*1e-4    

    
    
    s = time.time()
    
    control = solver.solveWrench(position,desired_wrench)
    
    control = [5,2,40,150]
    sphereControl = [5,40,90,150]
    sphereWrench = newModel.forwardPass(np.zeros((3,1)),sphereControl,True)
    cylinderWrench = newModel.forwardPass(np.zeros((3,1)),control)
    print(sphereWrench,cylinderWrench,sphereWrench-cylinderWrench)