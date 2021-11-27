#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 17:12:21 2021

@author: tabor
"""

import matplotlib.pyplot as plt
import numpy as np

noiseLevel = 5e-20
print('noise level ',noiseLevel)
noisylinearVelocity = linearVelocity + np.random.normal(0,np.max(linearVelocity)*noiseLevel,linearVelocity.shape)
noisyrotationalVelocity = rotationalVelocity + np.random.normal(0,np.max(rotationalVelocity)*noiseLevel,rotationalVelocity.shape)


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
from ll4ma_opt.solvers.line_search import NewtonMethod
optimizerParams = {'alpha':1,'ro':0.9,'c1':1e-9,'FP_PRECISION': 1e-15,'min_alpha': 1e-20}
prob = radiusSysid(np.diag((2e5,2e5,2e5,1,1,1))) # built with M(0.04) Q M(0.04)
optimizer = NewtonMethod(problem=prob,**optimizerParams)
s = time.time()
prob.updateVals(controls,env.magnets,positions,noisylinearVelocity,noisyrotationalVelocity)
solution, cost, converged, iterates = optimizer.optimize(prob.initial_solution,max_iterations=25)
print('opt time ', time.time()-s)
s = time.time()
xaxis, yaxis = prob.visualize_optimization(iterates,np.abs)
print('brute time ', time.time()-s)
print('solution ',solution)

import torch
x = torch.tensor(solution)
modelAccelFunc=lambda x: prob.forwardAcceleration(x, prob.dists, prob.dipoleStrength, prob.angles, prob.rhos, prob.zero)

modelAccel = modelAccelFunc(x)

modelAccelAverage = torch.mean(modelAccel,1)
deltaModeled =  modelAccelAverage.unsqueeze(1) - modelAccel

measuredAcelAverage = torch.mean(prob.accelerations,1)
deltaAccel=  measuredAcelAverage.unsqueeze(1) - prob.accelerations

deltaDifference = deltaAccel - deltaModeled
N = int(modelAccel.shape[1])
M = 6
P = 1
left = (deltaDifference.T @ prob.weights).unsqueeze(1)
right = deltaDifference.T.unsqueeze(2)
summedSquaredDelta = torch.sum(torch.bmm(left,right))
sigmaSquared = summedSquaredDelta.numpy()/(N*M-P)

bigJ = torch.swapaxes(torch.autograd.functional.jacobian(modelAccelFunc, x)[:,:,:,0],0,1)
batchedJtJ = torch.bmm(torch.swapaxes(bigJ, 1, 2),bigJ)
remap = torch.linalg.inv(torch.mean(batchedJtJ,0)).numpy()

covarianceParams = sigmaSquared*remap
print('covariance ', covarianceParams)