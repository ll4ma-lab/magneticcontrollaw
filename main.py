# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:27:35 2019

@author: tabor
"""
import numpy as np
from PDController import PDController
from discreteTimeSimulator import discreteTimeSimulator,simulateExperiment
from environments import PlanarProblem,SpaceProblem,PlanarEnvironments,SpaceEnvironments
from se3Utils import buildTF
import matplotlib.pyplot as plt
import time
import visualization
import DummyProfile
import SplineProfile
import TrapProfile
from Adaptive import Adaptive


from models import multiMagModel
from modelOptimization import multiMagSolver
from axisAlignedSolver import axisAlignedSolver

from ObjectProperties import ObjectProperties,boatProperties,sphereProperties

if __name__ == '__main__':

    useSpecialCaseSolver = False
    useSphere = True and not useSpecialCaseSolver 
    cheat = False
    velNoise = False
    controlNoise = False
    runAdaptive = True and useSphere
    
    env = SpaceProblem(SpaceEnvironments.CUBE)
    env.magnets = env.magnets[:]
    #env.trajectoryDimensions = 'xyz'

    simTime = np.sum(env.times)
    dt = 1
    numSteps = 500#int(simTime/dt)
    radius = 0.02
    mass,MOI = sphereProperties(8940,radius)
    
    sigma_copper = 5.8E+07
    sigma_al = 3.77E+07
    object_properties = ObjectProperties(mass=mass, MOI=MOI, radius=radius, sigma=sigma_copper )
    
    object_properties_approximate = ObjectProperties(mass=mass, MOI=MOI, radius=0.02, sigma=sigma_copper )
    
    boat_mass, boat_MOI = boatProperties(0.059,0.075,0.07,0.031)
    boat_properties = ObjectProperties(mass=boat_mass, MOI=boat_MOI)



    model = multiMagModel(env.magnets,object_properties,useSphere) #correct object properties for simulator
    solverModel = multiMagModel(env.magnets,object_properties_approximate,useSphere) # approximate object properties for controller
    
    if useSpecialCaseSolver:
        solver = axisAlignedSolver(solverModel, env.trajectoryDimensions)

    else:
        solver = multiMagSolver(solverModel,env.trajectoryDimensions,useSphere,10)

    controller = PDController(env.startState,0.005,0.000001,0.005,0.000002,0.0,0.000)
  
    
  
    simulator = discreteTimeSimulator(env.startState,dt,object_properties) #density, diamter
    velocityProfile = SplineProfile.SplineProfile(env,simTime,env.times)
    adaptive = Adaptive(env.magnets,'all', object_properties_approximate, 25)
    
    states,wrenches,desiredWrenchs,desiredStates,controls,minimizationCosts = simulateExperiment(simulator,controller,env,velocityProfile,model,solver,adaptive,numSteps=numSteps,dt=dt,cheat=cheat,runAdaptive=runAdaptive)
    
    
    print('Time spent doing forwad model ' + str(model.time))
    print('Time spent doing solving model ' + str(solver.time))
    print('Time spent doing velocity ' + str(velocityProfile.time))
    print('Time spent doing simulation ' + str(simulator.time))
    visStart = time.time()
    positions = np.array([state[0][:3,3] for state in states])
    linearVelocity = np.array([np.dot(state[0][:3,:3],state[2][:,0]) for state in states])
    rotationalVelocity = np.array([np.dot(state[0][:3,:3],state[1][:,0]) for state in states])
    transforms = np.array([state[0] for state in states])
    desiredFinalTF = buildTF(desiredStates[-1][:3,0],desiredStates[-1][3:6,0])
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
