# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:27:35 2019

@author: tabor
"""
import numpy as np
import gurobi
#from matlabModel import matlabModel
from pythonModel import pythonModel
#from matlabModel import matlabModel
from cylinderModel import cylinderSolver,cylinderModel
from PDController import PDController
from exponentialSimulator import exponentialSimulator, sphereProperties
from environments import PlanarProblem
import matplotlib.pyplot as plt
import time
import visualization
import DummyProfile
import SplineProfile
magnetLimit = 1





env = PlanarProblem()
simTime = np.sum(env.times)
dt = 1
numSteps = int(simTime/dt)
#model = matlabModel(env.magnets,False)
new_model = True
if(new_model):
    model = cylinderModel(env.magnets)
    solver = cylinderSolver(env.magnets)
    solver.setup(model,'xyt')

else:
    model = pythonModel(env.magnets,False)
    solver = gurobi.gurobiSolver(len(env.magnets)*3,magnetLimit)
    
controller = PDController(env.startState,0.003,0.00001,0.3,0.001)
simulator = exponentialSimulator(env.startState,dt,sphereProperties(8940,0.05)) #density, diamter
velocityProfile = SplineProfile.SplineProfile(env,simTime,env.times)
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
    #print(i)
#    if(i ==200):
#        break
    control =  solver.solveWrench(model,state,desired_wrench,env.trajectoryDimensions)
    controls.append(control)
    trueWrench = model.forwardPass(state,control)
    wrenches.append(trueWrench)
    state = simulator.simulate(trueWrench)
    states = np.concatenate((states,state),1)
    
    ControlOn = abs(np.array(control))>0.000000000001
    #print(np.where(ControlOn==1))
    numControl = ControlOn.sum()
    numControls.append(numControl)
runTime = time.time() - start
print(runTime)
print('Time spent doing model ' + str(model.time))
print('Time spent doing velocity ' + str(velocityProfile.time))

modified_controls = np.zeros((i,4))
for j in range(i):
    modified_controls[j,controls[j][0]] = controls[j][2] **2
npdesiredStates = np.array(desiredStates)
Plots = {'Position (m)':states[:2,:],
         'Orientation (radians)':states[5:6,:]*(180/np.pi),
         'Velocity (m/s)':states[6:8,:],
         'Control (A)': np.transpose(modified_controls[:,:]),

         "Force ":np.transpose(np.array(wrenches)[:,:3,0]),
         "Torque ":np.transpose(np.array(wrenches)[:,3:,0]),
         'Desired position (m)':np.transpose(npdesiredStates[:,:3,0]),
         'Desired velocity (m/s) ':np.transpose(npdesiredStates[:,6:9,0])}

visualization.plotStuff(Plots,'Controller')
fig = plt.figure()
fig.suptitle('Path (m)')
plt.plot(states[0,:],states[1,:])
plt.plot(npdesiredStates[:,0,0],npdesiredStates[:,1,0])
plt.show()
visualization.visualize2DPath(env,states,'Controller Path')
#visualization.animate2DPath(env,states,'Controller Path')

