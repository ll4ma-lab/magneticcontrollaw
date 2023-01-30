#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:20:02 2022

@author: tabor
"""
import sys

sys.path.append("..")  # Adds higher directory to python modules path.
import dataloader

import numpy as np
import time
import torch
from models import model_call_rotated_wrench, SphereModel, SphereControlCommand
from ObjectProperties import (
    ObjectProperties,
    build_mass_matrix,
    compute_sphere_properties,
    compute_boat_properties,
)

from solvers.modelOptimization import build_q, build_acceleration_q
from ll4ma_opt.problems import Problem, SteinWrapper
import matplotlib.pyplot as plt

from ll4ma_opt.solvers import NewtonMethod, GradientDescent, Momentum, BFGSMethod
from sysIDoptimization import RadiusSysid, SeperateRadiusSysid, MassSysID

import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from extract_sys_id import extract_sys_id_data

# dataloader.load('C:/Users/tabor/OneDrive/Documents/magneticcontrollaw/data/sim_2160s_2022-09-27 17.51.31.467776',globals())
# dataloader.load('/home/tabor/Documents/magneticcontrollaw/data/sim_5400s_2022-09-27 13.29.32.114902',globals())
# dataloader.load('/home/tabor/Documents/magneticcontrollaw/data/sim_2160s_2022-09-27 15.42.16.145535',globals())

(
    linear_velocity,
    rotational_velocity,
    positions,
    controls,
    initial_object_properties,
    boat,
    env,
) = extract_sys_id_data(
    "/media/tabor/ESD-ISO/data/updated/real_2023-01-25 23_45_58.252761"
)

magnets = env.magnets
sysID_type = RadiusSysid
optimizer_type = NewtonMethod
reset = False
use_stein = False
end_early = False
print_result = False
max_iterations = 40 if reset else 2


print("optimizer ", optimizer_type)
print("reset ", reset)

if optimizer_type is NewtonMethod:
    optimizerParams = {
        "alpha": 1,
        "rho": 0.9,
        "c1": 1e-9,
        "FP_PRECISION": 1e-10,
        "min_alpha": 1e-20,
    }
elif optimizer_type is GradientDescent or optimizer_type is Momentum:
    step = 1e-12
    optimizerParams = {
        "alpha": step,
        "rho": 0.9,
        "c1": 1e-9,
        "FP_PRECISION": 1e-10,
        "min_alpha": step * 1e-10,
    }
elif optimizer_type is BFGSMethod:
    optimizerParams = {
        "alpha": 1e-2,
        "rho": 0.9,
        "c1": 1e-9,
        "FP_PRECISION": 1e-10,
        "min_alpha": 1e-2,
    }
if optimizer_type is Momentum:
    optimizerParams["momentum"] = 0.9


prob = sysID_type(
    build_acceleration_q(env.trajectoryDimensions),
    object_properties=initial_object_properties,
    boat=boat,
)  # built with M(0.04) Q M(0.04)

stein_problem = SteinWrapper(prob, 25, 5)

if use_stein:
    optimizer = optimizer_type(problem=stein_problem, **optimizerParams)
else:
    optimizer = optimizer_type(problem=prob, **optimizerParams)


batch_size = 25
start_index = -batch_size
end_index = -1

results = []
initial_costs = []

for i in tqdm(range(1, len(controls) - batch_size)):

    start_index = i
    end_index = i + batch_size if len(controls) > i + batch_size else len(controls)

    selected_controls = controls[start_index - 1 : end_index - 1]
    selected_positions = positions[start_index - 1 : end_index - 1]
    selected_linear_velocities = linear_velocity[start_index - 1 : end_index - 1]
    selected_rotation_velocities = rotational_velocity[start_index - 1 : end_index - 1]

    prob.update_vals(
        selected_controls,
        env.magnets,
        selected_positions,
        selected_linear_velocities,
        selected_rotation_velocities,
        (0.0, 5.4),
    )
    # check cost with updated data but old parameters
    initial_costs.append(prob.cost(prob.initial_solution))

    result = optimizer.optimize(
        optimizer.problem.initial_solution, max_iterations=max_iterations
    )
    results.append(result)

    if not reset:
        optimizer.problem.initial_solution = result.solution
    if reset and optimizer_type is Momentum:
        optimizer.current_step *= 0
    if print_result:
        print(result.display())
    if end_early:
        break


shared_name = (
    str(optimizer_type)
    + "reset "
    + str(reset)
    + " multiplier "
    + str(prob.sigma_multiplier)
    + "\n"
)

if use_stein:
    costs = [stein_problem.cost(result.solution, False, True) for result in results]
    solutions = np.array(
        [
            result.solution.reshape((stein_problem.num_particles, prob.size()))
            for result in results
        ]
    )
else:
    costs = [result.cost for result in results]
    solutions = np.array([result.solution for result in results]).swapaxes(1, 2)

plt.figure()
plt.title(shared_name + "costs")
plt.plot(costs)


fig, axs = plt.subplots(stein_problem.size() if use_stein else prob.size(), 1)
plt.title(shared_name + "iterates over time")

for i in range(optimizer.problem.size()):
    for iterate in [result.iterates for result in results]:
        axs[i].plot(iterate[:, i])


fig, axs = plt.subplots(prob.size(), 1)
plt.title(shared_name + "solutions over time")
for i in range(prob.size()):
    for j in range(stein_problem.num_particles if use_stein else 1):
        axs[i].plot(solutions[:, j, i])

prob.update_vals(
    controls,
    env.magnets,
    positions,
    linear_velocity,
    rotational_velocity,
    (0.0, 5.4),
)

anim = prob.visualize_optimization(
    np.array([result.solution for result in results]),
    samples=100,
    show_costs=True,
    func=np.log,
)

anim_2d = prob.visualize_optimization(
    np.array([result.solution for result in results]),
    samples=2,
    show_costs=False,
    func=np.log,
)
last_anim_2d = prob.visualize_optimization(
    np.array(result.iterates), samples=2, show_costs=False, func=np.log
)

print("optimizer ", optimizer_type)
print("reset ", reset)
print(np.array([np.average(costs)]))
print("multiplier ", prob.sigma_multiplier)
