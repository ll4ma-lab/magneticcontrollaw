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


from sysIDoptimization import radiusSysid, SeperateRadiusSysid, MassSysID
from ll4ma_opt.solvers import (
    NewtonMethod,
    GradientDescent,
    GeneralizedGaussNewton,
    BFGSMethod,
    Momentum,
)
from models import SphereControlCommand
from solvers.modelOptimization import build_acceleration_Q
from ObjectProperties import ObjectProperties, boatProperties

from tqdm import tqdm

# dataloader.load("/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-15 15:42:21.459275",globals())
# dataloader.load("/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-17 17:30:14.948530",globals())


# funky trial
# dataloader.load("/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-18 12:41:50.186459",globals())


# the one to use
# dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-15 15:42:21.459275',globals())


# dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-20 20:24:30.748327',globals())

# dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-21 14:43:00.938421',globals())

# dataloader.load("/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-17 15:04:47.848272",globals())
# dataloader.load("/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/sim_500s_2022-01-15 14.45.40.028901",globals())

# dataloader.load("/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-19 15:25:19.938560",globals())


# square
# dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-21 18:48:19.207757',globals())
# dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-21 18:20:33.288123',globals())


# diamond
# dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-21 19:33:23.933468',globals())
# dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-21 21:10:15.576111',globals())


# square fixed orientation
# dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-21 22:08:22.671149',globals())

# rotate in place
# dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-22 00:52:58.546108',globals())


# dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-01-22 13:07:42.600754',globals())

# adaptive U
# dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/test_real_2022-03-24 17:00:48.946183',globals())

# dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/test_real_2022-03-25 13:25:01.501892',globals())


# dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-03-28 12:21:41.207511',globals())

# dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-03-28 18:22:30.645765',globals())


# dataloader.load('/home/tabor/catkin_ws/src/magnetic_system/MagneticControlLaw/data/real_2022-04-19 13:42:36.530973',globals())

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

# sphere
dataloader.load("/home/tabor/Downloads/real_2022-04-26 18_05_06.499346", globals())


# cuboid
# dataloader.load('/home/tabor/Downloads/real_2022-03-31 15_01_35.949906',globals())

# physical = True
import matplotlib.pyplot as plt


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
    boat_mass, boat_MOI = boatProperties(0.059, 0.075, 0.07, 0.031)
    boat_properties = ObjectProperties(
        mass=boat_mass, MOI=boat_MOI
    )  # + ObjectProperties(0.15,np.zeros(3))
    mass, MOI = mass
    initial_object_properties = ObjectProperties(mass, MOI, radius * 5, 58000000.0 * 5)

boat = boat_properties
velocity_shift = 3
start = -numSteps + 5
end = -5

# FIX incorrect velocity in real data
linearVelocity = np.array(
    [
        np.linalg.inv(state[0][:3, :3]) @ state[2][:, 0]
        for state in states[1 + start + velocity_shift : end - 1 + velocity_shift]
    ]
)
rotationalVelocity = np.array(
    [
        np.linalg.inv(state[0][:3, :3]) @ state[1][:, 0]
        for state in states[1 + start + velocity_shift : end - 1 + velocity_shift]
    ]
)

from scipy.spatial.transform import Rotation as R

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
sysID_type = radiusSysid
optimizer_type = NewtonMethod


if add_noise:
    noisylinearVelocity = linearVelocity + np.random.normal(
        0, 5e-5, linearVelocity.shape
    )
    noisyrotationalVelocity = rotationalVelocity + np.random.normal(
        0, 1e-3, rotationalVelocity.shape
    )
else:
    noisylinearVelocity = linearVelocity
    noisyrotationalVelocity = rotational_velocity


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
    build_acceleration_Q(env.trajectoryDimensions),
    object_properties=initial_object_properties,
    boat=boat,
)  # built with M(0.04) Q M(0.04)

optimizer = optimizer_type(problem=prob, **optimizerParams)
s = time.time()
prob.updateVals(
    some_controls,
    env.magnets,
    positions,
    noisylinearVelocity,
    noisyrotationalVelocity,
    (0.0, 5.4),
)

results = []
for i in tqdm(range(1)):
    initial_solution = np.random.uniform(prob.min_bounds, prob.max_bounds, (2, 1))
    result = optimizer.optimize(initial_solution, max_iterations=100)
    results.append(result)
print("opt time ", time.time() - s)


if sysID_type is radiusSysid:
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
    for i in range(20):
        iterate = np.stack(
            [
                result.iterates[i] if result.iterates.shape[0] > i else result.solution
                for result in results
            ]
        ).reshape((-1, 1))
        iterates.append(iterate)
    _ = prob.visualize_optimization(iterates, func=np.log, show_costs=True)

    print("brute time ", time.time() - s)
else:
    _ = prob.visualize_optimization(result.iterates, func=np.log)


print("solution ", result.solution)

prob.visualize_predicted_acceleration(result.solution)

import torch

x = torch.tensor(result.solution)
modelAccelFunc = lambda x: prob.forward_acceleration(
    prob.get_object_properties(x, torch),
    prob.dists,
    prob.dipoleStrength,
    prob.angles,
    prob.rhos,
    prob.zero,
)

modelAccel = modelAccelFunc(x)

modelAccelAverage = torch.mean(modelAccel, 1)
deltaModeled = modelAccelAverage.unsqueeze(1) - modelAccel

measuredAcelAverage = torch.mean(prob.accelerations, 1)
deltaAccel = measuredAcelAverage.unsqueeze(1) - prob.accelerations

deltaDifference = deltaAccel - deltaModeled
N = int(modelAccel.shape[1])
M = 3
P = prob.size()
left = (deltaDifference.T @ prob.weights).unsqueeze(1)
right = deltaDifference.T.unsqueeze(2)
summedSquaredDelta = torch.sum(torch.bmm(left, right))
sigmaSquared = summedSquaredDelta.numpy() / (N * M - P)

bigJ = torch.swapaxes(
    torch.autograd.functional.jacobian(modelAccelFunc, x)[:, :, :, 0], 0, 1
)
batchedJtJ = torch.bmm(torch.swapaxes(bigJ, 1, 2), bigJ)
remap = torch.linalg.inv(torch.mean(batchedJtJ, 0)).numpy()

covarianceParams = sigmaSquared * remap
print("covariance ", covarianceParams)
