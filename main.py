# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:27:35 2019

@author: tabor
"""
import numpy as np
from cylinderModel import cylinderSolver,cylinderModel
from PDController import PDController
from exponentialSimulator import  sphereProperties
from discreteTimeSimulator import discreteTimeSimulator
from environments import PlanarProblem,SpaceProblem
from se3Utils import buildTF
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
a=b

simTime = np.sum(env.times)
dt = 1
numSteps = int(simTime/dt)
diameter = 0.04
objectProperties = sphereProperties(8940,diameter)

model = cylinderModel(env.magnets,diameter)
solver = cylinderSolver(env.magnets)
solver.setup(model,env.trajectoryDimensions,objectProperties,env.tolerances)



controller = PDController(env.startState,0.005,0.00001,0.005,0.00001,0.0,0.000)

simulator = discreteTimeSimulator(env.startState,dt,objectProperties) #density, diamter


velocityProfile = SplineProfile.SplineProfile(env,simTime,env.times)

state = simulator.state
states = []
wrenches = []
desiredStates = []
controls = []

start = time.time()

for i in range(numSteps):

    desiredState = velocityProfile.getDesiredState(i*dt,state)
    desired_wrench,p,d = controller.getDesiredWrench(state,desiredState)
    #env.problem_specific_modification(state)
    control =  solver.solveWrench(model,state[0][:3,3],desired_wrench,env.trajectoryDimensions)
    trueWrench = model.forwardPass(state[0][:3,3],control)
    desiredStates.append(desiredState)
    #trueWrench = desired_wrench
    wrenches.append(trueWrench)
    controls.append(control)

    state = simulator.simulate(trueWrench)
    states.append(state)
    
runTime = time.time() - start
print(runTime)
print('Time spent doing forwad model ' + str(model.time))
print('Time spent doing solving model ' + str(solver.time))
print('Time spent doing velocity ' + str(velocityProfile.time))
print('Time spent doing simulation ' + str(simulator.time))

positions = np.array([state[0][:3,3] for state in states])
linearVelocity = np.array([np.dot(state[0][:3,:3],state[1][:,0]) for state in states])
rotationalVelocity = np.array([np.dot(state[0][:3,:3],state[2][:,0]) for state in states])
transforms = np.array([state[0] for state in states])
desiredFinalTF = buildTF(desiredState[:3,0],desiredState[3:6,0])
remainingError = np.array([controller.computeError(desiredFinalTF,state[0]) for state in states])

#visualization stuff
env.radius*=0.5
anim = visualization.animate2DPath(env,positions,transforms,controls,"")
npdesiredStates = np.array(desiredStates)

Plots = {'Position (m)':np.transpose(positions),
         'Velocity (m/s)':np.transpose(linearVelocity),
         "Force ":np.transpose(np.array(wrenches)[:,:3,0]),
         "Torque ":np.transpose(np.array(wrenches)[:,3:,0]),
         'Desired position (m)':np.transpose(npdesiredStates[:,:3,0]),
         'Desired velocity (m/s) ':np.transpose(npdesiredStates[:,6:9,0])}

visualization.plotStuff(Plots,'Controller')

Plots = {'Position Error(m)':np.abs(np.transpose(remainingError[:,:3,0])),
         'Orientation Error(radians)':np.abs(np.transpose(remainingError[:,3:6,0]))}

visualization.plotStuff(Plots,'Error')