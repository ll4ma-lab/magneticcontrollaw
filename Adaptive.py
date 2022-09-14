#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 13:02:06 2021

@author: tabor
"""
import sysIDoptimization
from ll4ma_opt.solvers import Momentum
from modelOptimization import buildQ
import numpy as np
import matplotlib.pyplot as plt
class Adaptive:
    def __init__(self, magnets, dimensions, object_properties, batch_size = 1,internal_sysID = sysIDoptimization.radiusSysid,boat=False):
        
        self.magnets = magnets
        self.object_properties = object_properties
        
        
        Q = buildQ(dimensions,2e3,1e0)
        print(Q)
        self.problem = internal_sysID(Q,boat=boat)
        
        step = 1e-12/ batch_size
        optimizerParams = {'alpha':step,'rho':0.9,'min_alpha': step/2} 
        
        optimizerParams = {'alpha':step,'rho':0.9,'c1':1e-9,'FP_PRECISION': 1e-15,'min_alpha': 1e-20, 'momentum':0.8}
        
        self.optimizer = Momentum(problem=self.problem,**optimizerParams)
        self.batch_size = batch_size
        
        self.linear_velocities = [] 
        self.rotational_velocities = []
        self.controls = []
        self.positions = []
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
        
        current_solution = np.array((self.object_properties.radius,self.object_properties.sigma,1)).reshape(((3,1)))
        if(len(self.parameters) == 0):
            self.parameters.append(current_solution)
        
        
        batch_size = min(self.batch_size,len(self.controls)-1)
        if batch_size < self.batch_size * 0.5:
            return
        
        selected_controls = self.controls[-batch_size-1:-1]
        selected_positions = np.array(self.positions[-batch_size :])
        selected_linear_velocities = np.array(self.linear_velocities[-batch_size:])
        selected_rotation_velocities = np.array(self.rotational_velocities[-batch_size:])
        

        self.problem.updateVals(selected_controls,
                                self.magnets,
                                selected_positions, 
                                selected_linear_velocities,
                                selected_rotation_velocities)
        
        result = self.optimizer.optimize(current_solution,max_iterations=1)
        
        updated_properties = self.problem.get_object_properties(result.solution,np)
        self.parameters.append(result.solution)

        self.object_properties.radius = updated_properties.radius
        self.object_properties.sigma = updated_properties.sigma
        self.object_properties.mass = updated_properties.mass
        self.object_properties.MOI = updated_properties.MOI

        
    def plot(self,true_vector):
        parameters = np.stack(self.parameters)
        fig,axs = plt.subplots(3,1)
        
        for i in range(3):
            axs[i].plot(parameters[:,i],'r')
            axs[i].plot([0,parameters.shape[0]],[true_vector[i],true_vector[i]])
        
        