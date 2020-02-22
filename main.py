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
from environments import PlanarProblem
import matplotlib.pyplot as plt
import time
import visualization
magnetLimit = 9


runTime = 1600
dt = 5
numSteps = int(runTime/dt)


env = PlanarProblem()
model = matlabModel(env.magnets)
solver = gurobi.gurobiSolver(len(env.magnets)*3,magnetLimit)
controller = PDController(env.startState,0.06,0.000007,0.5,0.0001)
simulator = exponentialSimulator(env.startState,dt,sphereProperties(8940,0.05)) #density, diamter

numControls = []
controls = []
state = env.startState
states = np.copy(state)
simulator.setState(state)
start = time.time()
for i in range(numSteps):
    desired_wrench = controller.getDesiredWrench(state,env.desiredState)
    A,B = model.getAB(state)
    control =  solver.solveWrench(A,B,desired_wrench)
    controls.append(control)
    trueWrench = model.forwardPass(state,control)
    #print(i)
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

Plots = {'Position (m)':states[:3,:],
         'Orientation (m)':states[3:6,:],
         'Velocity (m/s)':states[6:9,:],
         "Control Values":np.transpose(np.array(controls)[:,:,0])}

visualization.plotStuff(Plots)
fig = plt.figure()
fig.suptitle('Path (m)')
plt.plot(states[0,:],states[1,:])
plt.show()
