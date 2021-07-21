# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:27:35 2019

@author: tabor
"""
import numpy as np
from cylinderModel import cylinderSolver,cylinderModel
from PDController import PDController
from discreteTimeSimulator import discreteTimeSimulator,sphereProperties
from environments import PlanarProblem,SpaceProblem
from se3Utils import buildTF
import matplotlib.pyplot as plt
import time
import visualization
import DummyProfile
import SplineProfile
import TrapProfile


ll4maOpt = False

if ll4maOpt:
    from models import multiMagModel,multiMagSolver
if __name__ == '__main__':

    cheat = False
    velNoise = False
    controlNoise = False
    
    
    env = SpaceProblem(1)
    #env.trajectoryDimensions = 'xyz'
    
    simTime = np.sum(env.times)
    dt = 1
    numSteps = int(simTime/dt)
    diameter = 0.04
    objectProperties = sphereProperties(8940,diameter)
    
    if(ll4maOpt):
        model = multiMagModel(env.magnets,0.04)
        solver = multiMagSolver(model,env.trajectoryDimensions)
    else:
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
    minimizationCosts = []
    
    for i in range(numSteps):
    
        desiredState = velocityProfile.getDesiredState(i*dt,state)
        desired_wrench,p,d = controller.getDesiredWrench(state,desiredState)
       
        desiredStates.append(desiredState)

        if(cheat):
            trueWrench = desired_wrench
            control = [0,0,0,0]
        else:
            print(i)
            control =  solver.solveWrench(state[0][:3,3],desired_wrench)
            trueWrench = model.forwardPass(state[0][:3,3],control)
            wrenchError = trueWrench-desired_wrench
            cost = np.sum((wrenchError[:3,:]**2)) + 200*np.sum((wrenchError[3:,:]**2)) 
            minimizationCosts.append(cost)
        wrenches.append(trueWrench)
        controls.append(control)
       
        env.problem_specific_modification(state,trueWrench)
        state = simulator.simulate(trueWrench)
        states.append(state)
        
    runTime = time.time() - start
    print(runTime)
    print('Time spent doing forwad model ' + str(model.time))
    print('Time spent doing solving model ' + str(solver.time))
    print('Time spent doing velocity ' + str(velocityProfile.time))
    print('Time spent doing simulation ' + str(simulator.time))
    visStart = time.time()
    positions = np.array([state[0][:3,3] for state in states])
    linearVelocity = np.array([np.dot(state[0][:3,:3],state[2][:,0]) for state in states])
    rotationalVelocity = np.array([np.dot(state[0][:3,:3],state[1][:,0]) for state in states])
    transforms = np.array([state[0] for state in states])
    desiredFinalTF = buildTF(desiredState[:3,0],desiredState[3:6,0])
    remainingError = np.array([controller.computeError(desiredFinalTF,state[0]) for state in states])
    
    #visualization stuff
    env.radius*=0.5
    anim = visualization.animate2DPath(env,positions,transforms,controls,name="",savePlotSteps=velocityProfile.switchingTimes,save=False)
    npdesiredStates = np.array(desiredStates)
    
    Plots = {'Position (m)':np.transpose(positions),
             'Velocity (m/s)':np.transpose(linearVelocity),
             'Rot velocity (_/s)':np.transpose(rotationalVelocity),
             "Force ":np.transpose(np.array(wrenches)[:,:3,0]),
             "Torque ":np.transpose(np.array(wrenches)[:,3:,0]),
             'Desired position (m)':np.transpose(npdesiredStates[:,:3,0]),
             'Desired velocity (m/s) ':np.transpose(npdesiredStates[:,6:9,0])}
    
    visualization.plotStuff(Plots,'Controller')
    
    Plots = {'Position Error(m)':np.abs(np.transpose(remainingError[:,:3,0])),
             'Orientation Error(radians)':-np.transpose(remainingError[:,3:6,0])}
    
    visualization.plotStuff(Plots,'Error')
    print('Time spent doing visualization ' + str(time.time()-visStart))
    
    plt.figure()    
    plt.plot(minimizationCosts)

    import dataloader
    from datetime import datetime
    dataloader.save("data/sim_" + str(numSteps) + 's_' +  str(datetime.now()).replace(":","."),globals(),dir())
