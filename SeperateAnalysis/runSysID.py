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


from sysIDoptimization import radiusSysid
from ll4ma_opt.solvers.line_search import NewtonMethod,GradientDescent
from models import SphereControlCommand
from modelOptimization import build_acceleration_Q
from ObjectProperties import ObjectProperties,boatProperties
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


#dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-22 13:07:42.600754',globals())

#adaptive U
#dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/test_real_2022-03-24 17:00:48.946183',globals())

#dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/test_real_2022-03-25 13:25:01.501892',globals())


#dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-03-28 12:21:41.207511',globals())

#dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-03-28 18:22:30.645765',globals())


#dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-04-19 13:42:36.530973',globals())

#dataloader.load('/media/tabor/Storage/new_experiments/U/real_2022-04-01 17:06:41.504047',globals())
#dataloader.load('/media/tabor/Storage/new_experiments/U/real_2022-04-04 17:10:16.939314',globals())
#dataloader.load('/media/tabor/Storage/new_experiments/U/real_2022-04-04 18:33:05.822924',globals())
#dataloader.load('/media/tabor/Storage/new_experiments/U/real_2022-04-05 13:32:50.136111',globals())
#dataloader.load('/media/tabor/Storage/new_experiments/U/real_2022-04-05 17:30:39.811535',globals())


#square 
#dataloader.load('/media/tabor/Storage/new_experiments/square/real_2022-04-26 14:05:48.643385',globals())
#dataloader.load('/media/tabor/Storage/new_experiments/square/real_2022-04-26 15:03:52.189066',globals())
#dataloader.load('/media/tabor/Storage/new_experiments/square/real_2022-04-26 16:06:02.773579',globals())
#dataloader.load('/media/tabor/Storage/new_experiments/square/real_2022-04-26 17:03:51.259098',globals())
#dataloader.load('/media/tabor/Storage/new_experiments/square/real_2022-04-26 18:05:06.499346',globals())
dataloader.load('/home/tabor/Downloads/real_2022-04-26 18_05_06.499346',globals())


#physical = True
import matplotlib.pyplot as plt


if type(controls[0]) == list:
    controls = [SphereControlCommand(mag_num = control[0],
                                     dipole_strength = control[1],
                                     psi = control[2],
                                     xi = control[3]) for control in controls]
    boat_mass, boat_MOI = boatProperties(0.059,0.075,0.07,0.031)
    boat_properties = ObjectProperties(mass=boat_mass, MOI=boat_MOI) #+ ObjectProperties(0.15,np.zeros(3))
    mass,MOI = mass
    initial_object_properties = ObjectProperties(mass,MOI,radius,58000000.0)



start = -numSteps + 5
end = -5




linearVelocity = np.array([np.dot(state[0][:3,:3],state[2][:,0]) for state in states[1+start:end-1]])
rotationalVelocity = np.array([np.dot(state[0][:3,:3],state[1][:,0]) for state in states[1+start:end-1]])

positions = np.array([state[0][:3,3] for state in states[1+start:end-1]])

#linearVelocity =  (np.array([state[0][:3,3] for state in states[2+start:]])-  np.array([state[0][:3,3] for state in states[start+1:-1]]))/2
boat = boat_properties


some_controls = controls[start:end-2]


solutions = []

add_noise = False
magnets = env.magnets
sysID_type = radiusSysid
optimizer_type = NewtonMethod


if add_noise:
    noisylinearVelocity = linearVelocity + np.random.normal(0,5e-5,linearVelocity.shape)
    noisyrotationalVelocity = rotationalVelocity + np.random.normal(0,1e-3,rotationalVelocity.shape)
else:
    noisylinearVelocity = linearVelocity
    noisyrotationalVelocity = rotationalVelocity





optimizerParams = {'alpha':1,'rho':0.9,'c1':1e-3,'FP_PRECISION': 1e-10,'min_alpha': 1e-20}

prob = sysID_type(build_acceleration_Q(env.trajectoryDimensions),object_properties=initial_object_properties, boat = boat) # built with M(0.04) Q M(0.04)

optimizer = NewtonMethod(problem=prob,**optimizerParams)
s = time.time()
prob.updateVals(some_controls,env.magnets,positions,noisylinearVelocity,noisyrotationalVelocity,(0.0,54))
result = optimizer.optimize(prob.initial_solution,max_iterations=500)
solutions.append(result.solution[0,0])
print('opt time ', time.time()-s)

possible_r = np.linspace(prob.min_bounds[0],prob.max_bounds[0],100)[:,0]

possibleCosts = [prob.cost(np.stack((possible_r[i],prob.initial_solution[1,0])).reshape((2,1))) for i in range(len(possible_r))]
fig = plt.figure()
plt.plot(possible_r,np.abs(possibleCosts),'r')


print('opt time ', time.time()-s)
s = time.time()
_ = prob.visualize_optimization(result.iterates,np.abs)
print('brute time ', time.time()-s)
print('solution ',result.solution)

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