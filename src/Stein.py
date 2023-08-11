# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:33:58 2023

@author: tabor
"""

from sysIDoptimization import RadiusSysid
from ll4ma_opt.problems import SteinWrapper
from ObjectProperties import compute_sphere_properties, ObjectProperties
import numpy as np
import torch
from models import model_call_rotated_wrench
from matplotlib.animation import FuncAnimation


class SteinSysid(SteinWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.sigma_multiplier = self.problem.sigma_multiplier
    
    def update_vals(self, *args, **kwargs):
        self.problem.update_vals(*args, **kwargs)
        
    def get_object_properties(self, x, library=np):
        
        particles = x.reshape(-1, self.original_size, 1)


        masses = []
        MOIs = []

        for particle in particles:
            radius = particle[0,0]
            mass, MOI = compute_sphere_properties(self.problem.material_density, radius, library)
            masses.append(mass)
            MOIs.append(MOI)
        
        radii = particles[:,0].squeeze()
        sigmas = particles[:,1].squeeze()
        
        mass = np.stack(masses).squeeze()
        MOI = np.stack(MOIs).squeeze()

        object_properties = ObjectProperties(
            mass=mass, MOI=MOI, radius=radii, sigma=sigmas * self.problem.sigma_multiplier
        )
        return object_properties


from solvers.modelOptimization import SphericalAccelOptimization 

import matplotlib.pyplot as plt

class ExpectationControlOptimization(SphericalAccelOptimization):
    def __init__(self, weights):
        super().__init__(weights)
        self.mass_matrix_inv = torch.eye(6)

    def tensor_cost(self, x):
        selected_wrench = self.get_wrench(x)
        
        desired_matrix = self.desired.unsqueeze(1).expand((6,selected_wrench.shape[0],1))
        predicted_acceleration = torch.bmm(self.mass_matrix_inv,selected_wrench).transpose(0,1)
        error = 1e12 * (desired_matrix - predicted_acceleration)
        
        right =  (self.weights @ error.squeeze()).unsqueeze(2).transpose(0,1)
        left = error.transpose(0,1).transpose(1,2)
        
        cost = torch.sum(torch.bmm(left,right))
        return cost

    def tensor_error(self, x):
        #needs a sum across batch dimension
        selected_wrench = self.get_wrench(x)

        desired_matrix = self.desired.unsqueeze(1).expand((6,selected_wrench.shape[0],1))
        predicted_acceleration = torch.bmm(self.mass_matrix_inv,selected_wrench).transpose(0,1)
        error = 1e12 * (desired_matrix - predicted_acceleration)
        weighted_error = torch.sqrt(self.weights) @ error
        return weighted_error
    
    def get_wrench(self, x):
        particles = self.frho0.shape[0]
        
        dipole_strength = x[:1,0]
        angles = x[1:,0]

        dipole_strengths = dipole_strength.expand((particles))
        angles_array = angles.unsqueeze(1).expand((2,particles))
        
        rhos = self.rho.expand((3,particles))
        zeros = self.zero.expand(particles)
        return model_call_rotated_wrench(
            dipole_strengths,
            angles_array,
            (self.frho0, self.trho0, self.frho90, self.fphi90, self.ttheta90),
            rhos,
            library=torch,
            zero=zeros,
        )