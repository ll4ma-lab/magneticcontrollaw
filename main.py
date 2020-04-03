# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:27:35 2019

@author: tabor
"""
import numpy as np
import gurobi
#from matlabModel import matlabModel
from pythonModel import pythonModel
from PDController import PDController
from exponentialSimulator import exponentialSimulator, sphereProperties
from environments import PlanarProblem
import matplotlib.pyplot as plt
import time
import visualization
import DummyProfile
import SplineProfile
magnetLimit = 1


simTime = 900
dt = 1
numSteps = int(simTime/dt)


env = PlanarProblem()
#model = matlabModel(env.magnets,False)
model = pythonModel(env.magnets,False)

solver = gurobi.gurobiSolver(len(env.magnets)*3,magnetLimit)
controller = PDController(env.startState,0.003,0.000007,0.035,0.001)
simulator = exponentialSimulator(env.startState,dt,sphereProperties(8940,0.05)) #density, diamter
velocityProfile = SplineProfile.SplineProfile(env,simTime)
numControls = []
controls = []
wrenches = []
desiredStates = []
state = env.startState
states = np.copy(state)
simulator.setState(state)
start = time.time()

for i in range(numSteps):
    desiredState = velocityProfile.getDesiredState(i*dt)
    desired_wrench = controller.getDesiredWrench(state,desiredState)
    desiredStates.append(desiredState)
    state[2] = 0
    A,B = model.getAB(state)
    control =  solver.solveWrench(A,B,desired_wrench,env.trajectoryDimensions)
    controls.append(control)
    trueWrench = model.forwardPass(state,control)
    wrenches.append(trueWrench)

    state = simulator.simulate(trueWrench)
    states = np.concatenate((states,state),1)
    
    ControlOn = abs(control)>0.000000000001
    #print(np.where(ControlOn==1))
    numControl = ControlOn.sum()
    numControls.append(numControl)
runTime = time.time() - start
print(runTime)
print('Time spent doing model ' + str(model.time))
print('Time spent doing velocity ' + str(velocityProfile.time))

npdesiredStates = np.array(desiredStates)
Plots = {'Position (m)':states[:2,:],
         'Orientation (radians)':states[5:6,:],
         'Velocity (m/s)':states[6:8,:],
         "Force ":np.transpose(np.array(wrenches)[:,:3,0]),
         "Torque ":np.transpose(np.array(wrenches)[:,3:,0]),
         "Control Values":np.transpose(np.array(controls)[:,:,0]),
         'Desired position (m)':np.transpose(npdesiredStates[:,:3,0]),
         'Desired velocity (m/s) ':np.transpose(npdesiredStates[:,6:9,0])}

visualization.plotStuff(Plots,'Controller')
fig = plt.figure()
fig.suptitle('Path (m)')
plt.plot(states[0,:],states[1,:])
plt.plot(npdesiredStates[:,0,0],npdesiredStates[:,1,0])
plt.show()
visualization.visualize2DPath(env,states,'Controller Path')
visualization.animate2DPath(env,states,'Controller Path')

