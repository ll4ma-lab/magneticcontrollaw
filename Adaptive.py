#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 13:02:06 2021

@author: tabor
"""
import sysIDoptimization
from ll4ma_opt.solvers import Momentum
import numpy as np

class Adaptive:
    def __init__(self, magnets, object_properties, state, batch_size = 1):
        
        self.magnets = magnets
        self.object_properties = object_properties
        self.problem = sysIDoptimization.radiusSysid(np.diag((2e5,2e5,2e5,1,1,1)))
        
        step = 1
        optimizerParams = {'alpha':step,'ro':0.9,'min_alpha': step/2} 
        
        self.optimizer = Momentum(problem=self.problem,**optimizerParams)
        self.batch_size = batch_size
        
        self.linear_velocities = [] 
        self.rotational_velocities = []
        self.controls = []
        self.positions = []
        self.add_state(state)
        
        self.parameters = []
        
    def add_state(self,state):
        current_linear_velocity = state[0][:3,:3] @ state[2][:,0]
        current_rotational_velocity = state[0][:3,:3] @ state[1][:,0]
        
        self.linear_velocities.append(current_linear_velocity)
        self.rotational_velocities.append(current_rotational_velocity)
        self.positions.append(np.copy(state[0][:3,3]))


    def step(self, state, control):
        self.add_state(state)
        self.controls.append(control)
        
        
        batch_size = min(self.batch_size,len(self.controls))
        
        self.problem.updateVals(np.array(self.controls[-batch_size:]),
                                self.magnets,
                                np.array(self.positions[-batch_size - 1:-1]), 
                                np.array(self.linear_velocities[-batch_size-1:]),
                                np.array(self.rotational_velocities[-batch_size-1:]))
        
        current_solution = np.array(self.object_properties['radius']).reshape(((1,1)))
        
        solution, cost, converged, iterates = self.optimizer.optimize(current_solution,max_iterations=1) 
        self.parameters.append(solution[0,0])
        
        self.object_properties['radius'] = solution[0,0]
        