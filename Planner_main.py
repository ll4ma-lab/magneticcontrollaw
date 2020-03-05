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
from pythonModel import pythonModel
from matlabModel import matlabModel
from exponentialSimulator import exponentialSimulator, sphereProperties
from environments import PlanarProblem
import matplotlib.pyplot as plt
import time
import visualization
magnetLimit = 1
dt = 5
simTime = 50
numSteps = int(simTime/dt)


env = PlanarProblem()
#model = matlabModel(env.magnets)
model = pythonModel(env.magnets,False)

solver = gurobi.gurobiSolver(len(env.magnets)*3,magnetLimit)
simulator = exponentialSimulator(env.startState,dt,sphereProperties(8940,0.05)) #density, diamter

numControls = []

state = np.copy(env.startState)
simulator.setState(state)
start = time.time()

gurobi_model,info,plan_states,controls = solver.solvePlan(numSteps,model,env.desiredState,simulator,env.trajectoryDimensions)



states = np.copy(env.startState)

for i in range(numSteps):
    control = controls[i]
    trueWrench = model.forwardPass(state,control)
    state = simulator.simulate(trueWrench)
    states = np.concatenate((states,state),1)
    
    numControl = abs(control)>0.000000000001
    numControl = numControl.sum()
    numControls.append(numControl)

runTime = time.time() - start
print(runTime)
Plots = {'Position (m)':states[:3,:],
         'Orientation (radians)':states[3:6,:],
         'Position plan (m)':plan_states[:3,:],
         'Velocity (m/s)':states[6:9,:],
         "Control Values":np.transpose(np.array(controls)[:,:,0])}

visualization.plotStuff(Plots,'Planner')
fig = plt.figure()
fig.suptitle('Path (m)')
plt.plot(states[0,:],states[1,:])
plt.show()

visualization.visualize2DPath(env,states,'Planner Path')
