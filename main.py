# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:27:35 2019

@author: tabor
"""
import numpy as np
import gurobi
from matlabModel import matlabModel
from PDController import PDController
from exponentialSimulator import exponentialSimulator, sphereProperties
import matplotlib.pyplot as plt
import time
 
 
startState = np.zeros((12,1))
desiredState = np.zeros((12,1))
desiredState[:3,0] = [1,2,3]
desiredState = desiredState / 1000
dt = 0.5

All_magnets = 0.001 * np.array([(-130, 0, 0),( 0, 0, -130),
                   (0, -130, 0),( 0, 0, 130),
                   (130, 0, 0),( 0, 130, 0)])
All_magnets = All_magnets.tolist()

magnets = [All_magnets[0],All_magnets[1],All_magnets[2]]


model = matlabModel(magnets)
solver = gurobi.gurobiSolver(len(magnets)*3)
controller = PDController(startState,0.0007,0.0000007,0.025,0.000025)
simulator = exponentialSimulator(startState,dt,sphereProperties(8940,0.05)) #density, diamter


state = np.copy(startState)
states = np.copy(state)
simulator.setState(state)
randShape = np.random.randint(0,2,(6))
start = time.time()
for i in range(1000):
    #desired_wrench = randShape * np.cos(i*np.pi/50)/10
    desired_wrench = controller.getDesiredWrench(state,desiredState)
    A,B = model.getAB(state)
    control =  solver.solve(A,B,desired_wrench)
    trueWrench = model.forwardPass(state,control)
    state = simulator.simulate(trueWrench)
    states = np.concatenate((states,state),1)
runTime = time.time() - start
print(runTime)
for i in range(3):
    plt.plot(states[i,:])
plt.show()

for i in range(3):
    plt.plot(states[3+i,:])
plt.show()