#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:20:02 2022

@author: tabor
"""
import dataloader

import numpy as np
import time
import torch
from models import modelCallRotatedWrench,sphereModel,SphereControlCommand
from ObjectProperties import ObjectProperties,buildMassMatrix,sphereProperties,boatProperties

from modelOptimization import buildQ,build_acceleration_Q
from ll4ma_opt.problems import Problem,SteinWrapper
import matplotlib.pyplot as plt 

from ll4ma_opt.solvers import NewtonMethod,GradientDescent,Momentum,BFGSMethod
from sysIDoptimization import radiusSysid,SeperateRadiusSysid,MassSysID

import matplotlib.pyplot as plt
from tqdm import tqdm
#dataloader.load('C:/Users/tabor/OneDrive/Documents/magneticcontrollaw/data/sim_2160s_2022-09-27 17.51.31.467776',globals())
#dataloader.load('/home/tabor/Documents/magneticcontrollaw/data/sim_5400s_2022-09-27 13.29.32.114902',globals())
#dataloader.load('/home/tabor/Documents/magneticcontrollaw/data/sim_2160s_2022-09-27 15.42.16.145535',globals())

controls = env = adaptive = mass = MOI = radius = sigma_copper = numSteps = None



#dataloader.load('/home/tabor/Downloads/real_2022-10-10 15 54 23.074092',globals())
dataloader.load('C:/Users/tabor/Downloads/real_2022-04-27 17_35_24.878509',globals())

if type(adaptive.controls[0]) == list:
    adaptive.controls = [SphereControlCommand(mag_num = control[0],
                                     dipole_strength = control[1],
                                     psi = control[2],
                                     xi = control[3]) for control in adaptive.controls]
    boat_properties = ObjectProperties(mass=adaptive.problem.boat_mass, MOI=adaptive.problem.boat_MOI)
    mass,MOI = mass
boat = boat_properties

magnets = env.magnets
sysID_type = radiusSysid
optimizer_type = GradientDescent
reset = False
max_iterations = 100 if reset else 5


print('optimizer ' ,optimizer_type)
print('reset ', reset)

if optimizer_type is NewtonMethod:
    optimizerParams = {'alpha':1,'rho':0.9,'c1':1e-9,'FP_PRECISION': 1e-10,'min_alpha': 1e-20}
elif optimizer_type is GradientDescent or optimizer_type is Momentum:
    step = 1e-2
    optimizerParams = {'alpha':step,'rho':0.9,'c1':1e-9,'FP_PRECISION': 1e-10,'min_alpha': step}
elif optimizer_type is BFGSMethod:
    optimizerParams = {'alpha':1e-2,'rho':0.9,'c1':1e-9,'FP_PRECISION': 1e-10,'min_alpha': 1e-2}
if optimizer_type is Momentum:
    optimizerParams['momentum'] = 0.9




initial_object_properties=ObjectProperties(mass=mass, MOI=MOI, radius=radius, sigma=sigma_copper)

prob = sysID_type(build_acceleration_Q(env.trajectoryDimensions),object_properties=initial_object_properties, boat = boat) # built with M(0.04) Q M(0.04)

stein_problem = SteinWrapper(prob,20)


optimizer = optimizer_type(problem=stein_problem,**optimizerParams)


batch_size = 25
start_index = -batch_size
end_index = -1

results = []

for i in tqdm(range(1,numSteps-batch_size,1)):
    if i == 1000:
        break
        
    start_index = i
    end_index = i + batch_size if numSteps > i+batch_size else numSteps

    selected_controls = adaptive.controls[start_index-1:end_index-1]
    selected_positions = np.array(adaptive.positions[start_index :end_index])
    selected_linear_velocities = np.array(adaptive.linear_velocities[start_index:end_index])
    selected_rotation_velocities = np.array(adaptive.rotational_velocities[start_index:end_index])
    
    prob.updateVals(selected_controls,adaptive.magnets,selected_positions,selected_linear_velocities,selected_rotation_velocities,(0.61,5.4))
    
    
    
    result = optimizer.optimize(stein_problem.initial_solution,max_iterations=max_iterations)
    results.append(result)
    
    if not reset:
        stein_problem.initial_solution = result.solution
    if reset and optimizer_type is Momentum:
        optimizer.current_step *= 0
        
    
    print(result.display())
    break

shared_name = str(optimizer_type) + 'reset ' + str(reset) + ' multiplier ' + str(prob.sigma_multiplier) + '\n'

costs = [stein_problem.cost(result.solution,False,True) for result in results]
iterates = [result.iterates for result in results]
solutions = np.array([result.solution.reshape((stein_problem.num_particles,prob.size())) for result in results])
solutions = np.array([result.solution.reshape((stein_problem.num_particles,prob.size())) for result in results])

plt.figure()
plt.title(shared_name + 'costs')
plt.plot(costs)



fig,axs = plt.subplots(stein_problem.size(),1)
plt.title(shared_name + 'iterates over time')

for i in range(stein_problem.size()):
    for iterate in iterates:
        axs[i].plot(iterate[:,i])
            
        
fig,axs = plt.subplots(prob.size(),1)
plt.title(shared_name + 'solutions over time')
for i in range(prob.size()):
    for j in range(stein_problem.num_particles):
        axs[i].plot(solutions[:,j,i])
            


anim = prob.visualize_optimization( np.array([result.solution for result in results]),samples=100,show_costs=True)
anim_2d = prob.visualize_optimization( np.array([result.solution for result in results]),samples=2,show_costs=False)

print('optimizer ' ,optimizer_type)
print('reset ', reset)
print(np.array([np.average(costs)]))
print('multiplier ',prob.sigma_multiplier)