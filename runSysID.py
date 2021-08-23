#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 17:12:21 2021

@author: tabor
"""

import matplotlib.pyplot as plt
import numpy as np

accel = np.transpose(noisylinearVelocity[1:,:]-noisylinearVelocity[:-1,:])
force = np.transpose(np.array(wrenches)[:,:3,0])
torque = np.transpose(np.array(wrenches)[:,3:,0])
rotAccel = np.transpose(noisyrotationalVelocity[1:,:]-noisyrotationalVelocity[:-1,:])

division = force[:,1:]/accel

division2 = torque[:,1:]/rotAccel
accelPlots = {"accel" :accel,
             "Force ":force,
             "ratio":division,
             "rot accel":rotAccel,
             "Torque ": torque,
             "ratio 2":division2,
             'Desired accel ':np.transpose(npdesiredStates[:-1,6:9,0]-npdesiredStates[1:,6:9,0])}
    
visualization.plotStuff(accelPlots,'Controller')



from sysIDoptimization import radiusSysid
from solvers.line_search import NewtonMethod
optimizerParams = {'alpha':1,'ro':0.9,'c1':1e-9,'FP_PRECISION': 1e-5,'min_alpha': 1e-20}
prob = radiusSysid(np.diag((2e5,2e5,2e5,1,1,1))) # built with M(0.04) Q M(0.04)
optimizer = NewtonMethod(problem=prob,**optimizerParams)
s = time.time()
prob.updateVals(controls,env.magnets,positions,noisylinearVelocity,noisyrotationalVelocity)
solution, cost, converged, iterates = optimizer.optimize(prob.initial_solution,max_iterations=25)
print('opt ', time.time()-s)
s = time.time()
prob.visualize_optimization(iterates)
print('brute ', time.time()-s)
print(solution)