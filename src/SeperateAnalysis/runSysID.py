#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 17:12:21 2021

@author: tabor
"""
import sys

sys.path.append("..")  # Adds higher directory to python modules path.
import numpy as np


import dataloader
import numpy as np
import time
from scipy.spatial.transform import Rotation as R


from sysIDoptimization import RadiusSysid, SeperateRadiusSysid, MassSysID
from ll4ma_opt.solvers import (
    NewtonMethod,
    GradientDescent,
    GeneralizedGaussNewton,
    BFGSMethod,
    Momentum,
)
from models import SphereControlCommand
from solvers.modelOptimization import build_acceleration_q
from ObjectProperties import ObjectProperties, compute_boat_properties

from tqdm import tqdm

# dataloader.load('/media/tabor/Storage/new_experiments/U/real_2022-04-01 17:06:41.504047',globals())
# dataloader.load('/media/tabor/Storage/new_experiments/U/real_2022-04-04 17:10:16.939314',globals())
# dataloader.load('/media/tabor/Storage/new_experiments/U/real_2022-04-04 18:33:05.822924',globals())
# dataloader.load('/media/tabor/Storage/new_experiments/U/real_2022-04-05 13:32:50.136111',globals())
# dataloader.load('/media/tabor/Storage/new_experiments/U/real_2022-04-05 17:30:39.811535',globals())


# square
# dataloader.load('/media/tabor/Storage/new_experiments/square/real_2022-04-26 14:05:48.643385',globals())
# dataloader.load('/media/tabor/Storage/new_experiments/square/real_2022-04-26 15:03:52.189066',globals())
# dataloader.load('/media/tabor/Storage/new_experiments/square/real_2022-04-26 16:06:02.773579',globals())
# dataloader.load('/media/tabor/Storage/new_experiments/square/real_2022-04-26 17:03:51.259098',globals())
# dataloader.load('/media/tabor/Storage/new_experiments/square/real_2022-04-26 18:05:06.499346',globals())

states = controls = env = i = None

# sphere
#dataloader.load("/home/tabor/Downloads/real_2022-04-26 18_05_06.499346", globals())


#new data
#dataloader.load("/home/tabor/data/real_2023-01-24 16_39_19.923278",globals())
dataloader.load("/media/tabor/ESD-ISO/data/75%/real_2023-01-25 17_55_58.639885",globals())
#dataloader.load("/media/tabor/ESD-ISO/data/50%/real_2023-01-25 18_54_49.649503",globals())

# cuboid
# dataloader.load('/home/tabor/Downloads/real_2022-03-31 15_01_35.949906',globals())

# physical = True
import matplotlib.pyplot as plt

transform_modify_function = lambda x: x


if type(controls[0]) == list:
    controls = [
        SphereControlCommand(
            mag_num=control[0],
            dipole_strength=control[1],
            psi=control[2],
            xi=control[3],
        )
        for control in controls
    ]
    boat_mass, boat_MOI = compute_boat_properties(0.059, 0.075, 0.07, 0.031)
    boat_properties = ObjectProperties(
        mass=boat_mass, MOI=boat_MOI
    )  # + ObjectProperties(0.15,np.zeros(3))
    mass, MOI = mass
    initial_object_properties = ObjectProperties(mass, MOI, radius * 5, 58000000.0 * 5)
    
    transform_modify_function = np.linalg.inv

try:
    num_steps = numSteps
except:
    num_steps = i

try:
    initial_object_properties
except:
    initial_object_properties = object_properties_approximate

boat = boat_properties
velocity_shift = 0
start = -num_steps + 5
end = -5
alt_rotational_velocity = False

# FIX incorrect velocity in real data
linear_velocity = np.array(
    [
        transform_modify_function(state[0][:3, :3]) @ state[2][:, 0]
        for state in states[1 + start + velocity_shift : end - 1 + velocity_shift]
    ]
)
rotational_velocity = np.array(
    [
        transform_modify_function(state[0][:3, :3]) @ state[1][:, 0]
        for state in states[1 + start + velocity_shift : end - 1 + velocity_shift]
    ]
)

if alt_rotational_velocity:
    angles = np.array(
        [
            R.from_matrix(state[0][:3, :3]).as_euler("xyz")
            for state in states[1 + start : end]
        ]
    )
    rotational_velocity = []
    for current, future in zip(angles[:-1, 2], angles[1:, 2]):
        if np.abs(current - future) > np.pi:
            current -= np.sign(current) * 2 * np.pi
        vel = future - current
        rotational_velocity.append([0, 0, vel])
    rotational_velocity = np.array(rotational_velocity)

positions = np.array([state[0][:3, 3] for state in states[1 + start : end - 1]])
some_controls = controls[start : end - 2]


add_noise = False
magnets = env.magnets
sysID_type = RadiusSysid
optimizer_type = NewtonMethod


if add_noise:
    noisy_linear_velocity = linear_velocity + np.random.normal(
        0, 5e-5, linear_velocity.shape
    )
    noisy_rotational_velocity = rotational_velocity + np.random.normal(
        0, 1e-3, rotational_velocity.shape
    )
else:
    noisy_linear_velocity = linear_velocity
    noisy_rotational_velocity = rotational_velocity


if optimizer_type is NewtonMethod:
    optimizerParams = {
        "alpha": 1,
        "rho": 0.9,
        "c1": 1e-3,
        "FP_PRECISION": 1e-16,
        "min_alpha": 1e-20,
    }
elif optimizer_type is GradientDescent or optimizer_type is Momentum:
    step = 1e-11
    optimizerParams = {
        "alpha": step,
        "rho": 0.9,
        "c1": 1e-11,
        "FP_PRECISION": 1e-16,
        "min_alpha": step * 1e-20,
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
    build_acceleration_q('xyt'),
    object_properties=initial_object_properties,
    boat=boat,
)  # built with M(0.04) Q M(0.04)

optimizer = optimizer_type(problem=prob, **optimizerParams)
s = time.time()
prob.update_vals(
    some_controls,
    env.magnets,
    positions,
    noisy_linear_velocity,
    noisy_rotational_velocity,
    (0.0, 5.4),
)

results = []
for i in tqdm(range(10)):
    initial_solution = np.random.uniform(prob.min_bounds, prob.max_bounds, (prob.size(), 1))
    result = optimizer.optimize(initial_solution, max_iterations=40)
    results.append(result)
print("opt time ", time.time() - s)

if sysID_type is RadiusSysid:
    possible_r = np.linspace(prob.min_bounds[0], prob.max_bounds[0], 1000)[:, 0]

    possibleCosts = [
        prob.cost(
            np.stack((possible_r[i], prob.initial_solution[1, 0])).reshape((2, 1))
        )
        for i in range(len(possible_r))
    ]
    fig = plt.figure()
    plt.plot(possible_r, np.log(possibleCosts), "r")

    s = time.time()

    iterates = []
    for i in range(40):
        iterate = np.stack(
            [
                result.iterates[i] if result.iterates.shape[0] > i else result.solution
                for result in results
            ]
        ).reshape((-1, 1))
        iterates.append(iterate)
    _ = prob.visualize_optimization(iterates, func=np.log, show_costs=True,save_snapshots=False)

    print("brute time ", time.time() - s)
else:
    _ = prob.visualize_optimization(result.iterates, func=np.log)

solution = results[np.argmin([result.cost for result in results])].solution

print("solution ", result.solution)

prob.visualize_predicted_acceleration(solution)


#positions = np.array([state[0][:3, 3] for state in states])
# linear_velocity = np.array(
#     [np.dot(state[0][:3, :3], state[2][:, 0]) for state in states]
# )
# rotational_velocity = np.array(
#     [np.dot(state[0][:3, :3], state[1][:, 0]) for state in states]
# )
import visualization
plots = {
    "Position (m)": np.transpose(positions),
    "Velocity (m/s)": np.transpose(linear_velocity),
    "Rot velocity (_/s)": np.transpose(rotational_velocity),

}

visualization.plot_stuff(plots, "Controller")

