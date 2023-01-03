#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:20:02 2022

@author: tabor
"""
import sys
sys.path.append("..") # Adds higher directory to python modules path.
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
from scipy.spatial.transform import Rotation as R

#dataloader.load('C:/Users/tabor/OneDrive/Documents/magneticcontrollaw/data/sim_2160s_2022-09-27 17.51.31.467776',globals())
#dataloader.load('/home/tabor/Documents/magneticcontrollaw/data/sim_5400s_2022-09-27 13.29.32.114902',globals())
#dataloader.load('/home/tabor/Documents/magneticcontrollaw/data/sim_2160s_2022-09-27 15.42.16.145535',globals())

controls = env = adaptive = mass = MOI = radius = sigma_copper = numSteps = None



#dataloader.load('/home/tabor/Downloads/real_2022-10-10 15 54 23.074092',globals())


#sphere
dataloader.load('/home/tabor/Downloads/real_2022-04-26 18_05_06.499346',globals())

#cuboid
#dataloader.load('/home/tabor/Downloads/real_2022-03-31 15_01_35.949906',globals())


if type(controls[0]) == list:
    controls = [SphereControlCommand(mag_num = control[0],
                                     dipole_strength = control[1],
                                     psi = control[2],
                                     xi = control[3]) for control in controls]
    boat_mass, boat_MOI = boatProperties(0.059,0.075,0.07,0.031)
    boat_properties = ObjectProperties(mass=boat_mass, MOI=boat_MOI) #+ ObjectProperties(0.15,np.zeros(3))
    mass,MOI = mass
    initial_object_properties = ObjectProperties(mass,MOI,radius,58000000.0)
boat = boat_properties

magnets = env.magnets
sysID_type = radiusSysid
optimizer_type = NewtonMethod
reset = False
use_stein = False
end_early = False
print_result = False
max_iterations = 100 if reset else 5


print('optimizer ' ,optimizer_type)
print('reset ', reset)

if optimizer_type is NewtonMethod:
    optimizerParams = {'alpha':1,'rho':0.9,'c1':1e-9,'FP_PRECISION': 1e-10,'min_alpha': 1e-20}
elif optimizer_type is GradientDescent or optimizer_type is Momentum:
    step = 1e-11
    optimizerParams = {'alpha':step,'rho':0.9,'c1':1e-9,'FP_PRECISION': 1e-10,'min_alpha': step*1e-10}
elif optimizer_type is BFGSMethod:
    optimizerParams = {'alpha':1e-2,'rho':0.9,'c1':1e-9,'FP_PRECISION': 1e-10,'min_alpha': 1e-2}
if optimizer_type is Momentum:
    optimizerParams['momentum'] = 0.9




initial_object_properties=ObjectProperties(mass=mass, MOI=MOI, radius=radius, sigma=sigma_copper)

prob = sysID_type(build_acceleration_Q(env.trajectoryDimensions),object_properties=initial_object_properties, boat = boat) # built with M(0.04) Q M(0.04)

stein_problem = SteinWrapper(prob,20)

if use_stein:
    optimizer = optimizer_type(problem=stein_problem,**optimizerParams)
else:
    optimizer = optimizer_type(problem=prob,**optimizerParams)



batch_size = 25
start_index = -batch_size
end_index = -1

results = []
velocity_shift=3




boat = boat_properties
velocity_shift = 3
start = -numSteps + 5
end = -5


linearVelocity = np.array([np.linalg.inv(state[0][:3,:3])@state[2][:,0] for state in states[1+start+velocity_shift:end-1 + velocity_shift]])
rotationalVelocity =np.array([np.linalg.inv(state[0][:3,:3])@state[1][:,0] for state in states[1+start+velocity_shift:end-1 + velocity_shift]])

from scipy.spatial.transform import Rotation as R
angles = np.array([R.from_matrix(state[0][:3,:3]).as_euler('xyz') for state in states[1+start :end ]])
rotational_velocity = []
for current,future in zip(angles[:-1,2],angles[1:,2]):
    if np.abs(current-future) > np.pi:
        current -= np.sign(current) * 2 * np.pi    
    vel = future - current
    rotational_velocity.append([0,0,vel])
rotational_velocity = np.array(rotational_velocity)

positions = np.array([state[0][:3,3] for state in states[1+start:end-1]])
alligned_controls = controls[start:end-2]




for i in tqdm(range(1,numSteps-batch_size,1)):
    if i == 1000:
        break
        
    start_index = i
    end_index = i + batch_size if numSteps > i+batch_size else numSteps

    if adaptive:
        selected_controls = adaptive.controls[start_index-1:end_index-1]
        selected_positions = np.array(adaptive.positions[start_index :end_index])
        selected_linear_velocities = np.array(adaptive.linear_velocities[start_index:end_index])
        selected_rotation_velocities = np.array(adaptive.rotational_velocities[start_index:end_index])
    else:
        selected_controls = alligned_controls[start_index-1:end_index-1]
        selected_positions = positions[start_index-1:end_index-1]
        selected_linear_velocities = linearVelocity[start_index-1:end_index-1]
        selected_rotation_velocities = rotational_velocity[start_index-1:end_index-1]
        
        
    prob.updateVals(selected_controls,env.magnets,selected_positions,selected_linear_velocities,selected_rotation_velocities,(0.0,5.4))
    
    
    
    result = optimizer.optimize(optimizer.problem.initial_solution ,max_iterations=max_iterations)
    results.append(result)
    
    if not reset:
        optimizer.problem.initial_solution = result.solution
    if reset and optimizer_type is Momentum:
        optimizer.current_step *= 0
    if print_result:
        print(result.display())
    if end_early:
        break


shared_name = str(optimizer_type) + 'reset ' + str(reset) + ' multiplier ' + str(prob.sigma_multiplier) + '\n'

if use_stein:
    costs = [stein_problem.cost(result.solution,False,True) for result in results]
    solutions = np.array([result.solution.reshape((stein_problem.num_particles,prob.size())) for result in results])
else:
    costs = [result.cost for result in results]
    solutions = np.array([result.solution for result in results]).swapaxes(1, 2)

plt.figure()
plt.title(shared_name + 'costs')
plt.plot(costs)



fig,axs = plt.subplots(stein_problem.size(),1)
plt.title(shared_name + 'iterates over time')

for i in range(optimizer.problem.size()):
    for iterate in [result.iterates for result in results]:
        axs[i].plot(iterate[:,i])
            
        
fig,axs = plt.subplots(prob.size(),1)
plt.title(shared_name + 'solutions over time')
for i in range(prob.size()):
    for j in range(stein_problem.num_particles if use_stein else 1):
        axs[i].plot(solutions[:,j,i])
            
prob.updateVals(alligned_controls,env.magnets,positions,linearVelocity,rotational_velocity,(0.0,5.4))

anim = prob.visualize_optimization( np.array([result.solution for result in results]),samples=100,show_costs=True,func=np.log)

anim_2d = prob.visualize_optimization( np.array([result.solution for result in results]),samples=2,show_costs=False,func=np.log)
last_anim_2d = prob.visualize_optimization( np.array(result.iterates),samples=2,show_costs=False,func=np.log)

print('optimizer ' ,optimizer_type)
print('reset ', reset)
print(np.array([np.average(costs)]))
print('multiplier ',prob.sigma_multiplier)