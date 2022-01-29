#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 17:12:21 2021

@author: tabor
"""
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import numpy as np


import dataloader
import numpy as np
import time 
#dataloader.load("/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-15 15:42:21.459275",globals())
#dataloader.load("/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-17 17:30:14.948530",globals())


#funky trial
#dataloader.load("/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-18 12:41:50.186459",globals())


#the one to use
#dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-15 15:42:21.459275',globals())


#dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-20 20:24:30.748327',globals())

#dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-21 14:43:00.938421',globals())

#dataloader.load("/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-17 15:04:47.848272",globals())
#dataloader.load("/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/sim_500s_2022-01-15 14.45.40.028901",globals())

#dataloader.load("/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-19 15:25:19.938560",globals())




#square
#dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-21 18:48:19.207757',globals())
#dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-21 18:20:33.288123',globals())


#diamond
#dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-21 19:33:23.933468',globals())
#dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-21 21:10:15.576111',globals())


#square fixed orientation
#dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-21 22:08:22.671149',globals())

#rotate in place
#dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-22 00:52:58.546108',globals())


dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-22 13:07:42.600754',globals())


physical = True
import matplotlib.pyplot as plt



start = -1050
end = -850




linearVelocity = np.array([np.dot(state[0][:3,:3],state[2][:,0]) for state in states[1+start:end-1]])
rotationalVelocity = np.array([np.dot(state[0][:3,:3],state[1][:,0]) for state in states[1+start:end-1]])

positions = np.array([state[0][:3,3] for state in states[1+start:end-1]])

#linearVelocity =  (np.array([state[0][:3,3] for state in states[2+start:]])-  np.array([state[0][:3,3] for state in states[start+1:-1]]))/2


some_controls = controls[start:end-2]


solutions = []

add_noise = False

if add_noise:
    noisylinearVelocity = linearVelocity + np.random.normal(0,5e-5,linearVelocity.shape)
    noisyrotationalVelocity = rotationalVelocity + np.random.normal(0,1e-3,rotationalVelocity.shape)
else:
    noisylinearVelocity = linearVelocity
    noisyrotationalVelocity = rotationalVelocity




from sysIDoptimization import radiusSysid
from ll4ma_opt.solvers.line_search import NewtonMethod,GradientDescent
optimizerParams = {'alpha':5e-3/(abs(start-end)),'ro':0.9,'c1':1e-3,'FP_PRECISION': 1e-5,'min_alpha': 1e-20}
prob = radiusSysid(np.diag((2e3,2e3,0,0,0,1e0)),physical=physical) # built with M(0.04) Q M(0.04)
optimizer = GradientDescent(problem=prob,**optimizerParams)
s = time.time()
prob.updateVals(some_controls,env.magnets,positions,noisylinearVelocity,noisyrotationalVelocity)
solution, cost, converged, iterates = optimizer.optimize(prob.initial_solution,max_iterations=50)
solutions.append(solution[0,0])
print('opt time ', time.time()-s)


possible_r = np.linspace(prob.boundConstraints[0][0],prob.boundConstraints[0][1],100)

possibleCosts = [prob.cost(np.stack((possible_r[i],prob.initial_solution[1,0],prob.initial_solution[2,0])).reshape((3,1))) for i in range(len(possible_r))]
fig = plt.figure()
plt.plot(possible_r,np.log(possibleCosts),'r')


print('opt time ', time.time()-s)
s = time.time()
_ = prob.visualize_optimization(iterates,np.log)
print('brute time ', time.time()-s)
print('solution ',solution)

a=b

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