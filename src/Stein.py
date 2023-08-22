# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:33:58 2023

@author: tabor
"""

from sysIDoptimization import RadiusSysid
from ll4ma_opt.problems import SteinWrapper
from ObjectProperties import compute_sphere_properties, ObjectProperties, batch_sphere_properties
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

        radii = particles[:, 0, 0]
        sigmas = particles[:, 1, 0] * self.problem.sigma_multiplier

        object_properties = batch_sphere_properties(self.problem.material_density, radii, sigmas)
        return object_properties
