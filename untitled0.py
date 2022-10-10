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
from models import modelCallRotatedWrench,sphereModel
from ObjectProperties import ObjectProperties,buildMassMatrix,sphereProperties

from modelOptimization import buildQ,build_acceleration_Q
from ll4ma_opt.problems import Problem 
import matplotlib.pyplot as plt 

from ll4ma_opt.solvers.line_search import NewtonMethod,GradientDescent
from sysIDoptimization import radiusSysid,SeperateRadiusSysid,MassSysID



use_torch = False
use_GPU = False
def do_nothing(x):
    return x
def make_tensor(x):
    return torch.DoubleTensor(x).to('cuda')


#dataloader.load('C:/Users/tabor/OneDrive/Documents/magneticcontrollaw/data/sim_2160s_2022-09-27 17.51.31.467776',globals())
#dataloader.load('/home/tabor/Documents/magneticcontrollaw/data/sim_5400s_2022-09-27 13.29.32.114902',globals())
#dataloader.load('/home/tabor/Documents/magneticcontrollaw/data/sim_2160s_2022-09-27 15.42.16.145535',globals())
magnets = env.magnets
sysID_type = MassSysID
boat = controller_boat

initial_object_properties=ObjectProperties(mass=mass, MOI=MOI, radius=radius, sigma=sigma_copper)
optimizerParams = {'alpha':1,'rho':0.9,'c1':1e-9,'FP_PRECISION': 1e-18,'min_alpha': 1e-20}
prob = sysID_type(build_acceleration_Q(env.trajectoryDimensions),object_properties=initial_object_properties, boat = boat) # built with M(0.04) Q M(0.04)

if use_torch and use_GPU:
    prob.make_tensor = make_tensor
    prob.weights = make_tensor(prob.weights)

batch_size = 500

selected_controls = adaptive.controls[-batch_size-1:-1]
selected_positions = np.array(adaptive.positions[-batch_size :])
selected_linear_velocities = np.array(adaptive.linear_velocities[-batch_size:])
selected_rotation_velocities = np.array(adaptive.rotational_velocities[-batch_size:])


adaptive.problem.updateVals(selected_controls,
                        adaptive.magnets,
                        selected_positions, 
                        selected_linear_velocities,
                        selected_rotation_velocities)



prob.updateVals(selected_controls,adaptive.magnets,selected_positions,selected_linear_velocities,selected_rotation_velocities)


if use_torch:
    prob.initial_solution = prob.make_tensor(prob.initial_solution)
    prob.min_bounds = prob.make_tensor(prob.min_bounds)
    prob.max_bounds = prob.make_tensor(prob.max_bounds)
    prob.make_tensor = do_nothing 
    prob.unmake_tensor = do_nothing
    optimizer = NewtonMethod(problem=prob,**optimizerParams,library=torch)
    optimizer.cholesky_factorize = compute_inverse_cholesky_fast
else:
    optimizer = NewtonMethod(problem=prob,**optimizerParams)



result = optimizer.optimize(prob.initial_solution,max_iterations=150)
print(result.display())



prob.visualize_optimization(result.iterates,np.abs)
