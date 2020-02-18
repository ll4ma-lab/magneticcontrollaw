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
UnitConversionFactor = 10**6
magnetLimit = 1
startState = np.zeros((12,1))
desiredState = np.zeros((12,1))
desiredState[:3,0] = [1,2,0]
desiredState = UnitConversionFactor*desiredState / 10
runTime = 30
dt = 1
numSteps = int(runTime/dt)

All_magnets = 0.001 * np.array([(-80, 0, 0),( 0, 0, -80),
                   (0, -80, 0),( 0, 0, 80),
                   (80, 0, 0),( 0, 80, 0)])
All_magnets = All_magnets.tolist()

magnets = [All_magnets[0],All_magnets[1],All_magnets[2]]


model = matlabModel(magnets)
solver = gurobi.gurobiSolver(len(magnets)*3,magnetLimit)
controller = PDController(startState,0.06/UnitConversionFactor,0.000007/UnitConversionFactor,0.5/UnitConversionFactor,0.0001/UnitConversionFactor)
simulator = exponentialSimulator(startState,dt,sphereProperties(8940,0.05),UnitConversionFactor) #density, diamter

numControls = []
controls = []
state = np.copy(startState)
states = np.copy(state)
simulator.setState(state)
randShape = np.random.randint(0,2,(6))
start = time.time()
for i in range(numSteps):
    desired_wrench = controller.getDesiredWrench(state,desiredState)
    A,B = model.getAB(state)
    control =  solver.solveWrench(A,B,desired_wrench)
    controls.append(control)
    trueWrench = model.forwardPass(state,control)
    print(i)
    #print(trueWrench)
    #print(control)
    state = simulator.simulate(trueWrench)
    states = np.concatenate((states,state),1)
    
    ControlOn = abs(control)>0.000000000001
    print(np.where(ControlOn==1))
    numControl = ControlOn.sum()
    numControls.append(numControl)
runTime = time.time() - start
print(runTime)

fig = plt.figure()
fig.suptitle('Position (m)')
for i in range(3):
    plt.plot(states[i,:]/UnitConversionFactor)
plt.show()

fig = plt.figure()
fig.suptitle('Path (m)')
plt.plot(states[0,:]/UnitConversionFactor,states[1,:]/UnitConversionFactor)
plt.show()

for i in range(3):
    plt.plot(states[3+i,:]/UnitConversionFactor)
plt.show()

fig = plt.figure()
fig.suptitle('Non-zero control inputs')
plt.ylim(0,9)
plt.scatter(range(numSteps),numControls)
plt.show()

np_controls = np.array(controls)
fig.suptitle('')
for i in range(9):
    plt.plot(np_controls[:,i,0])
plt.show()