# -*- coding: utf-8 -*-
""" 
Created on Tue May 11 14:06:39 2021 
 
@author: Griffin Tabor 
"""
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np

import sys

sys.path.append("..")  # Adds higher directory to python modules path.
import matplotlib.pyplot as plt
import time
from PDController import PDController
from torch.autograd import Variable

from environments import PlanarProblem, SpaceProblem, SpaceEnvironments
import SplineProfile
from discreteTimeSimulator import DiscreteTimeSimulator
from ObjectProperties import compute_sphere_properties, ObjectProperties

from models import MultiMagModel
from solvers.modelOptimization import MultiMagSolver


env = SpaceProblem(SpaceEnvironments.AXIS)
controller = PDController(env.start_state, 0.005, 0.00001, 0.005, 0.00001, 0.0, 0.000)
velocityProfile = SplineProfile.SplineProfile(env, env.times)


radius = 0.02
sigma_copper = 5.87e07
mass, MOI = compute_sphere_properties(8940, radius)
object_properties = ObjectProperties(
    mass=mass, MOI=MOI, radius=radius, sigma=sigma_copper
)

if __name__ == "__main__":
    position = np.zeros((3, 1))
    model = MultiMagModel(env.magnets, object_properties, True)
    solver = MultiMagSolver(model, "all", True)
    # np.random.seed(473)
    for i in range(5):
        desired_wrench = (np.random.random((6, 1)) - 0.5) * 1e-4
        s = time.time()

        control = solver.solve_wrench(position, desired_wrench)
        print(time.time() - s)
        if time.time() - s > 1 and i > 0:
            break
