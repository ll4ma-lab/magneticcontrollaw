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
        super().__init__(*args, **kwargs)
        self.sigma_multiplier = self.problem.sigma_multiplier

    def update_vals(self, *args, **kwargs):
        self.problem.update_vals(*args, **kwargs)

    def get_object_properties(self, x, library=np):
        particles = x.reshape(-1, self.original_size, 1)

        masses = []
        MOIs = []

        for particle in particles:
            radius = particle[0, 0]
            mass, MOI = compute_sphere_properties(
                self.problem.material_density, radius, library
            )
            masses.append(mass)
            MOIs.append(MOI)

        radii = particles[:, 0].squeeze()
        sigmas = particles[:, 1].squeeze()

        mass = np.stack(masses).squeeze()
        MOI = np.stack(MOIs).squeeze()

        object_properties = ObjectProperties(
            mass=mass,
            MOI=MOI,
            radius=radii,
            sigma=sigmas * self.problem.sigma_multiplier,
        )
        return object_properties
