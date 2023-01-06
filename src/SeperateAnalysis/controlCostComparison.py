#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 12:31:38 2021

@author: tabor
"""
from models import MultiMagModel
from modelOptimization import multiMagSolver
from environments import PlanarProblem, SpaceProblem
from tqdm import tqdm


import sys

sys.path.append("..")  # Adds higher directory to python modules path.
import dataloader
import numpy as np
import time

dataloader.load(
    "/home/tabor/Documents/magneticcontrollaw/data/sim_2700s_2022-01-13 15.48.50.855692",
    globals(),
)
import matplotlib.pyplot as plt


doBrute = False
import time

if __name__ == "__main__":
    model = 5
    sphereCosts = []
    cylinderCosts = []
    bruteSphereCosts = []

    radius = 0.02
    mass = sphereProperties(8940, radius)
    object_properties = {"mass": mass[0], "MOI": mass[1], "radius": radius}

    sphereModel = MultiMagModel(env.magnets, object_properties, True)
    sphereSolver = multiMagSolver(sphereModel, env.trajectoryDimensions, True, 10)

    cylinderModel = MultiMagModel(env.magnets, object_properties, False)
    cylinderSolver = multiMagSolver(cylinderModel, env.trajectoryDimensions, False, 10)

    sphereTime = []
    cylinderTime = []

    for desired_wrench, state in tqdm(
        zip(desiredWrenchs[:500], states[:500]), total=len(states[:500])
    ):

        s = time.time()
        control1 = sphereSolver.solveWrench(state[0][:3, 3], desired_wrench)
        sphereTime.append(time.time() - s)
        trueWrench1 = sphereModel.forward_pass(state[0][:3, 3], control1)
        wrenchError1 = trueWrench1 - desired_wrench
        cost = np.sum((wrenchError1[:3, :] ** 2)) + 200 * np.sum(
            (wrenchError1[3:, :] ** 2)
        )
        sphereCosts.append(cost)

        s = time.time()
        control2 = cylinderSolver.solveWrench(state[0][:3, 3], desired_wrench)
        cylinderTime.append(time.time() - s)
        trueWrench2 = cylinderModel.forward_pass(state[0][:3, 3], control2)
        wrenchError2 = trueWrench2 - desired_wrench
        cost2 = np.sum((wrenchError2[:3, :] ** 2)) + 200 * np.sum(
            (wrenchError2[3:, :] ** 2)
        )
        cylinderCosts.append(cost2)

        if doBrute:
            control3 = sphereSolver.bruteSphereSolver(state[0][:3, 3], desired_wrench)
            trueWrench3 = sphereModel.forward_pass(state[0][:3, 3], control3)
            wrenchError3 = trueWrench3 - desired_wrench
            cost3 = np.sum((wrenchError3[:3, :] ** 2)) + 200 * np.sum(
                (wrenchError3[3:, :] ** 2)
            )
            bruteSphereCosts.append(cost3)
        else:
            bruteSphereCosts.append(0.0)

    plt.figure()
    plt.plot(np.array(sphereCosts) - np.array(bruteSphereCosts), "r")
    plt.plot(np.array(cylinderCosts) - np.array(bruteSphereCosts), "b")

    plt.figure()
    plt.plot(np.array(sphereCosts) / np.array(cylinderCosts))
    print(np.max(sphereTime[1:]), np.max(cylinderTime[1:]))

    fig, axes = plt.subplots(1, 1, figsize=(2, 8))

    axes.boxplot([sphereTime[1:]], positions=[1])
    axes.boxplot([cylinderTime[1:]], positions=[2])
