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
import visualization
UnitConversionFactor = 10**6 # change length unit used internally. 1 being meters 10**3 being milimeters
magnetLimit = 1
startState = np.zeros((12,1))
desiredState = np.zeros((12,1))
desiredState[:3,0] = [3,-1,0]
desiredState = UnitConversionFactor*desiredState
runTime = 100
dt = 1
numSteps = int(runTime/dt)


All_magnets = 0.001 * np.array([(-130, 0, 0),( 0, 0, -130),
                   (0, -130, 0),( 0, 0, 130),
                   (130, 0, 0),( 0, 130, 0)])
All_magnets = All_magnets.tolist()

magnets = [All_magnets[0],All_magnets[1],All_magnets[2]]


model = matlabModel(magnets)

solver = gurobi.gurobiSolver(len(magnets)*3,magnetLimit)

simulator = exponentialSimulator(startState,dt,sphereProperties(8940,0.05),UnitConversionFactor) #density, diamter
numControls = []

state = np.copy(startState)
simulator.setState(state)
start = time.time()

gurobi_model,info,plan_states = solver.solvePlan(numSteps,model,desiredState,simulator,"xyt")
controls = []
for i in range(numSteps):
    control_sym = info[i][1]
    On_OFF_sym = info[i][2]

    control = np.zeros(control_sym.shape)
    for j in range(control_sym.shape[0]):
        control[j,0]=control_sym[j,0].x * On_OFF_sym[j,0].x
    controls.append(control)



state = np.copy(startState)
states = np.copy(state)
simulator.setState(state)

for i in range(numSteps):
    control = controls[i]
    trueWrench = model.forwardPass(state,control)
    state = simulator.simulate(trueWrench)
    states = np.concatenate((states,state),1)
    
    numControl = abs(control)>0.000000000001
    numControl = numControl.sum()
    numControls.append(numControl)


Plots = {'Position (m)':states[:3,:]/UnitConversionFactor,
         'Position plan (m)':plan_states[:3,:]/UnitConversionFactor,
         'Velocity (m/s)':states[6:9,:]/UnitConversionFactor,
         "Control Values":np.transpose(np.array(controls)[:,:,0])}

visualization.plotStuff(Plots)
fig = plt.figure()
fig.suptitle('Path (m)')
plt.plot(states[0,:]/UnitConversionFactor,states[1,:]/UnitConversionFactor)
plt.show()

