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


#dataloader.load('/home/tabor/Documents/magneticcontrollaw/data/sim_5400s_2022-09-27 13.29.32.114902',globals())
dataloader.load('/home/tabor/Documents/magneticcontrollaw/data/sim_2160s_2022-09-27 15.42.16.145535',globals())
magnets = env.magnets
sysID_type = MassSysID
boat = controller_boat

initial_object_properties=ObjectProperties(mass=mass_initial, MOI=MOI_initial, radius=initial_radius_guess, sigma=sigma_copper)
optimizerParams = {'alpha':1,'rho':0.5,'c1':1e-9,'FP_PRECISION': 1e-9,'min_alpha': 1e-20}
prob = sysID_type(build_acceleration_Q(env.trajectoryDimensions)**(0.3),object_properties=initial_object_properties, boat = boat) # built with M(0.04) Q M(0.04)
prob.updateVals(controls,magnets,positions,linearVelocity,rotationalVelocity)

optimizer = NewtonMethod(problem=prob,**optimizerParams)
result = optimizer.optimize(prob.initial_solution,max_iterations=500)
print(result.solution, np.array([result.cost]))



prob.visualize_optimization(result.iterates,np.abs)
