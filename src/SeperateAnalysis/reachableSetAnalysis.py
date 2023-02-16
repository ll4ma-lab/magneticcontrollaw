#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:47:32 2021

@author: tabor
"""

import sys

sys.path.append("..")  # Adds higher directory to python modules path.

import numpy as np
from PDController import PDController
from discreteTimeSimulator import (
    DiscreteTimeSimulator,
    sphereProperties,
    simulate_experiment,
)
from environments import PlanarProblem, SpaceProblem
from se3Utils import build_tf
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

import time
import visualization
import DummyProfile
import SplineProfile
import TrapProfile
from tqdm import tqdm

useSphere = False

from models import MultiMagModel
from modelOptimization import multiMagSolver

if __name__ == "__main__":

    env = SpaceProblem(6)
    env.magnets = env.magnets[:-3]
    env.trajectory_dimensions = "xyz"

    simTime = np.sum(env.times)
    dt = 1
    numSteps = int(simTime / dt)
    radius = 0.02
    mass = sphereProperties(8940, radius)
    objectProperties = {"mass": mass[0], "MOI": mass[1], "radius": radius}

    model = MultiMagModel(env.magnets, objectProperties, useSphere)
    solverModel = MultiMagModel(env.magnets, objectProperties, useSphere)
    solver = multiMagSolver(solverModel, env.trajectory_dimensions, useSphere, 10)

    modifier = 10
    controller = PDController(
        env.start_state,
        0.005 * modifier,
        0.000002 * modifier,
        0.005 * modifier,
        0.000002 * modifier,
        0.0,
        0.000,
    )

    sampledVals = np.linspace(-3e-2, 3e-2, 16)
    x, y, z = np.meshgrid(sampledVals, sampledVals, sampledVals, indexing="ij")
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    samplePoints = np.stack((x, y, z)).T

    posErrorsPercent = []
    velErrors = []
    s = time.time()
    for i in tqdm(range(samplePoints.shape[0])):
        env.desired_state[:3, 0] = samplePoints[i, :]
        simulator = DiscreteTimeSimulator(
            env.start_state, dt, objectProperties
        )  # density, diamter
        velocityProfile = SplineProfile.SplineProfile(env, simTime, env.times)
        (
            states,
            wrenches,
            desiredWrenchs,
            desiredStates,
            controls,
            minimizationCosts,
        ) = simulate_experiment(
            simulator,
            controller,
            env,
            velocityProfile,
            model,
            solver,
            numSteps=numSteps,
            cheat=False,
            dt=dt,
            debug_prints=False,
        )

        velError = np.linalg.norm(np.dot(states[-1][0][:3, :3], states[-1][2][:, 0]))
        velErrors.append(velError)

        desiredFinalTF = build_tf(desiredStates[-1][:3, 0], desiredStates[-1][3:6, 0])
        remainingError = controller.compute_error(desiredFinalTF, states[-1][0])
        posError = np.linalg.norm(remainingError[:3])
        startingError = controller.compute_error(desiredFinalTF, np.eye(4))
        startingPosError = np.linalg.norm(startingError[:3])
        posErrorsPercent.append(posError / startingPosError)

    threshold = 0.1
    colors = [
        (1, 0, 0, 1 - error / threshold if error < threshold else 0)
        for error in posErrorsPercent
    ]
    print(time.time() - s)
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.scatter(samplePoints[:, 0], samplePoints[:, 1], samplePoints[:, 2], c=colors)
    ax.set_xlim(*env.visualization_boundary_y)
    ax.set_ylim(*env.visualization_boundary_y)
    ax.set_zlim(*env.visualization_boundary_y)
    visualization.draw_cubes(env.magnets, [], ax, env.radius)

    name = "sphere_" if useSphere else "cylinder_"
    import dataloader
    import dataloader
    from datetime import datetime

    dataloader.save(
        "../data/"
        + name
        + "_mags_"
        + str(len(env.magnets))
        + "_samples_"
        + str(samplePoints.shape[0])
        + str(datetime.now()).replace(":", "."),
        globals(),
        dir(),
    )
