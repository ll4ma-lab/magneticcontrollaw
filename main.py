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
from devinSolver import devinSolver

from ObjectProperties import ObjectProperties,boatProperties,sphereProperties
from sysIDoptimization import radiusSysid,SeperateRadiusSysid,MassSysID

if __name__ == '__main__':

    useSpecialCaseSolver = False
    useSphere = True
    cheat = False
    runAdaptive = False and useSphere
    inverseDynamics = False and useSphere and not cheat
    
    simulate_boat = False

    controllerKnowsBoat = True and simulate_boat

    velNoise = False
    controlNoise = False
    
    env = SpaceProblem(SpaceEnvironments.BACKANDFORTH)
    env.magnets = env.magnets[:]
    #env.trajectoryDimensions = 'xyz'

    sim_time = np.sum(env.times)

    
    sigma_copper = 5.87E+07
    sigma_al = 3.77E+07
    
    #simulator object
    radius = 0.02
    mass,MOI = sphereProperties(8940,radius)
    object_properties = ObjectProperties(mass=mass, MOI=MOI, radius=radius, sigma=sigma_copper )
    
    #controller object
    initial_radius_guess = 0.02
    mass_initial,MOI_initial = sphereProperties(8940,radius)
    object_properties_approximate = ObjectProperties(mass=mass_initial, MOI=MOI_initial, radius=initial_radius_guess, sigma=sigma_copper )
    
    #boat object
    boat_mass, boat_MOI = boatProperties(0.059,0.075,0.07,0.031)
    boat_properties = ObjectProperties(mass=boat_mass, MOI=boat_MOI)
    
    adaptive_model = MassSysID
    
    if simulate_boat:
        object_properties = object_properties + boat_properties
    
    if controllerKnowsBoat:
        controller_boat = boat_properties
    else: 
        controller_boat = None



    model = multiMagModel(env.magnets,object_properties,useSphere) #correct object properties for simulator
    solverModel = multiMagModel(env.magnets,object_properties_approximate,useSphere) # approximate object properties for controller
    
    if useSpecialCaseSolver:
        solver = axisAlignedSolver(solverModel, env.trajectoryDimensions)
        solver = devinSolver(solverModel)
    else:
        solver = multiMagSolver(model=solverModel,
                                stateCostString=env.trajectoryDimensions,
                                object_properties=object_properties_approximate,
                                sphereModel=useSphere,
                                inverse_dynamics=inverseDynamics,
                                boat=controller_boat,
                                numProcesses=10)
    
    if inverseDynamics:
        controller = PDController(env.startState,1.2,0.004,1.2,0.004,0,0)
    elif(useSpecialCaseSolver):
        controller = PDController(env.startState,0.0005,0.0,0.1,0.0,0.0,0.000)
    else:
        controller = PDController(env.startState,0.005,0.000001,0.005,0.000002,0.0,0.000)
    simulator = discreteTimeSimulator(env.startState,object_properties) 
    velocityProfile = SplineProfile.SplineProfile(env,sim_time,env.times)
    adaptive = Adaptive(env.magnets,'all', object_properties_approximate, 25,adaptive_model, boat_properties)
    
    states,wrenches,desiredWrenchs,desiredStates,controls,minimizationCosts = simulateExperiment(simulator,controller,env,velocityProfile,model,solver,adaptive,sim_time = sim_time,cheat=cheat,runAdaptive=runAdaptive,inverse_dynamics=inverseDynamics)
    
    
    #print('Time spent doing forwad model ' + str(model.time))
    #print('Time spent doing solving model ' + str(solver.time))
    #print('Time spent doing velocity ' + str(velocityProfile.time))
    #print('Time spent doing simulation ' + str(simulator.time))
    visStart = time.time()
    positions = np.array([state[0][:3,3] for state in states])
    linearVelocity = np.array([np.dot(state[0][:3,:3],state[2][:,0]) for state in states])
    rotationalVelocity = np.array([np.dot(state[0][:3,:3],state[1][:,0]) for state in states])
    transforms = np.array([state[0] for state in states])
    desiredFinalTF = buildTF(desiredStates[-1][:3,0],desiredStates[-1][3:6,0])
    remainingError = np.array([controller.computeError(transforms[0],state[0]) for state in states])
    
    
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
    
    
    Plots = {'desired Position(m)':np.array(desiredStates)[:,:3,0].T,
             'desired Orientation (radians)':np.array(desiredStates)[:,3:6,0].T}
    
    visualization.plotStuff(Plots,'Desired Pose')
    
    #print('Time spent doing visualization ' + str(time.time()-visStart))
    
    plt.figure()    
    plt.plot(minimizationCosts)
    
    diff = npdesiredStates[:,0,0] - positions[:,0]
    plt.figure()
    plt.plot(diff)
    print(np.amax(diff))