# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:36:54 2020

@author: tabor
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:27:35 2019

@author: tabor
"""
import numpy as np
import gurobi
from matlabModel import matlabModel
from exponentialSimulator import exponentialSimulator, sphereProperties
import matplotlib.pyplot as plt
import time
 
magnetLimit = 1
startState = np.zeros((12,1))
desiredState = np.zeros((12,1))
desiredState[:3,0] = [1,2,3]
desiredState = desiredState / 1000
runTime = 20
dt = 0.5
numSteps = int(runTime/dt)


All_magnets = 0.001 * np.array([(-130, 0, 0),( 0, 0, -130),
                   (0, -130, 0),( 0, 0, 130),
                   (130, 0, 0),( 0, 130, 0)])
All_magnets = All_magnets.tolist()

magnets = [All_magnets[0],All_magnets[1],All_magnets[2]]


model = matlabModel(magnets)

solver = gurobi.gurobiSolver(len(magnets)*3,magnetLimit)

simulator = exponentialSimulator(startState,dt,sphereProperties(8940,0.05)) #density, diamter
numControls = []

state = np.copy(startState)
simulator.setState(state)

start = time.time()

gurobi_model,info = solver.solvePlan(numSteps,model,desiredState,simulator)
controls = []
for i in range(numSteps):
    data = info[i][1]
    control = np.zeros(data.shape)
    for j in range(data.shape[0]):
        control[j,0]=data[j,0].x
    controls.append(control)

runTime = time.time() - start
print(runTime)

state = np.copy(startState)
states = np.copy(state)
simulator.setState(state)

for i in range(numSteps):
    control = controls[i]
    trueWrench = control#model.forwardPass(state,control)
    state = simulator.simulate(trueWrench)
    states = np.concatenate((states,state),1)
    
    numControl = abs(control)>0.000000000001
    numControl = numControl.sum()
    numControls.append(numControl)
runTime = time.time() - start
print(runTime)

fig = plt.figure()
fig.suptitle('Position (m)')
for i in range(3):
    plt.plot(states[i,:])
plt.show()

for i in range(3):
    plt.plot(states[3+i,:])
plt.show()

fig = plt.figure()
fig.suptitle('Non-zero control inputs')
plt.ylim(0,9)
plt.scatter(range(numSteps),numControls)