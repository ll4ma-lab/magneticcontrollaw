# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:27:35 2019

@author: tabor
"""
import numpy as np
from cylinderModel import cylinderSolver,cylinderModel
from PDController import PDController
from exponentialSimulator import exponentialSimulator, sphereProperties
from environments import PlanarProblem,SpaceProblem
import matplotlib.pyplot as plt
import time
import visualization
import DummyProfile
import SplineProfile
import TrapProfile
cheat = False
velNoise = False
controlNoise = False


env = SpaceProblem(3)
#env = PlanarProblem()
#env.times = (np.array(env.times) * 1.5).tolist()
if velNoise:
    env.startState[6:8,0] = (np.random.random((2)) - 0.5) * 0.005


simTime = np.sum(env.times)
dt = 1
numSteps = int(simTime/dt)
diameter = 0.04
objectProperties = sphereProperties(8940,diameter)

model = cylinderModel(env.magnets,diameter)
solver = cylinderSolver(env.magnets)
solver.setup(model,env.trajectoryDimensions,objectProperties,env.tolerances)



controller = PDController(env.startState,0.005,0.00004,0.005,0.00004,0.0,0.000)
#controller = PDController(env.startState,0.003,0.00001,0.3,0.001)
simulator = exponentialSimulator(env.startState,dt,objectProperties) #density, diamter
velocityProfile = SplineProfile.SplineProfile(env,simTime,env.times)
#velocityProfile = TrapProfile.TrapProfile(env,simTime,env.times)

controls = []
wrenches = []
pComponents = []
dComponents = []
desiredStates = []
state = env.startState
states = np.copy(state)
simulator.setState(state)
start = time.time()
modeledForces = []
desiredForces = []
for i in range(numSteps):
    controlState = np.copy(state)
    if velNoise:
        controlState[6:12,0] *= (np.random.random((2))*2 + 0.5)
    
    desiredState = velocityProfile.getDesiredState(i*dt,controlState)
    desired_wrench,p,d = controller.getDesiredWrench(controlState,desiredState)
    pComponents.append(p)
    dComponents.append(d)
    desiredStates.append(desiredState)
    env.problem_specific_modification(state)

    if(cheat):
        control = [0,0,0,0]
        maxForce = 0.00001
        desired_wrench[np.where(desired_wrench > maxForce)] = maxForce
        desired_wrench[np.where(desired_wrench < -maxForce)] = -maxForce
        trueWrench = desired_wrench
    else:
       control =  solver.solveWrench(model,state,desired_wrench,env.trajectoryDimensions)
       trueWrench = model.forwardPass(state,control)
    modeledForces.append(trueWrench)
    desiredForces.append(desired_wrench)
    if controlNoise:
        trueWrench[:2,0] += (np.random.random(2) - 0.5) * 0.00015
    controls.append(control) 

    wrenches.append(trueWrench)
    state = simulator.simulate(trueWrench)
    states = np.concatenate((states,state),1)
    
runTime = time.time() - start
print(runTime)
print('Time spent doing forwad model ' + str(model.time))
print('Time spent doing solving model ' + str(solver.time))
print('Time spent doing velocity ' + str(velocityProfile.time))

modified_controls = np.zeros((i,len(env.magnets)))
forceError = np.zeros((i,3))
npmodelF = np.array(modeledForces)
npdesiredF = np.array(desiredForces)
for j in range(i):
    modified_controls[j,controls[j][0]] = controls[j][2]
    edims = 2
    forceError[j,0] = np.arccos(np.dot(modeledForces[j][:edims,0],desiredForces[j][:edims,0])
            /(np.linalg.norm(desiredForces[j][:edims,0])*np.linalg.norm(modeledForces[j][:edims,0])))
    forceError[j,1] = np.linalg.norm(modeledForces[j][:edims]) / np.linalg.norm(desiredForces[j][:edims])
    forceError[j,2] = np.linalg.norm(modeledForces[j][:edims]-desiredForces[j][:edims])

npdesiredStates = np.array(desiredStates)
Plots = {'Position (m)':states[:3,:],
         'Orientation (degrees)':states[3:6,:]*(180/np.pi),
         'Velocity (m/s)':states[6:9,:],
         'Control (Am^2)': np.transpose(modified_controls[:,:]),
         "Force ":np.transpose(np.array(wrenches)[:,:3,0]),
         "Torque ":np.transpose(np.array(wrenches)[:,3:,0]),
         'Desired position (m)':np.transpose(npdesiredStates[:,:3,0]),
         'Desired velocity (m/s) ':np.transpose(npdesiredStates[:,6:9,0])}

visualization.plotStuff(Plots,'Controller')

errorPlots = {'Position error (m)':np.transpose(npdesiredStates[:,:3,0])-states[:3,:-1],
         'Orientation error (degrees)':(np.transpose(npdesiredStates[:,3:6,0]) -states[3:6,:-1])*(180/np.pi)}

visualization.plotStuff(errorPlots,'Controller Error')

forceErrorPlots = {'force angle error (degrees)':np.transpose(forceError[:,0:1] * (180/np.pi)),
         'force magnitude ratio':np.transpose(forceError[:,1:2]),
         'force error magnitude':np.transpose(forceError[:,2:3])}

visualization.plotStuff(forceErrorPlots,'Controller Force Stats')

fig = plt.figure()
fig.suptitle('Path (m)')
plt.plot(states[0,:],states[1,:],c='orange')
plt.plot(npdesiredStates[:,0,0],npdesiredStates[:,1,0])
plt.show()
env.radius*=0.5
visualization.visualize2DPath(env,states,'Controller Path')

anim = visualization.animate2DPath(env,states,'',save=True)

print('Time spent doing visualization ' + str(time.time() - start - runTime))

import dataloader
from datetime import datetime
dataloader.save("data/sim_" +str(datetime.now()).replace(":","."),globals(),dir())


