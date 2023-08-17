# -*- coding: utf-8 -*-
""" 
Created on Tue May 11 14:06:39 2021 
 
@author: Griffin Tabor 
"""
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"
from ll4ma_opt.problems import Problem
import numpy as np

import torch
import torch.multiprocessing as multiprocessing
from torch.autograd import Variable

from ll4ma_opt.solvers.line_search import (
    GradientDescent,
    NewtonMethod,
    NewtonMethodWolfeLine,
)
import matplotlib.pyplot as plt
from matplotlib import animation
import time

import sys

sys.path.append("..")  # Adds higher directory to python modules path.

from PDController import PDController

from environments import PlanarProblem, SpaceProblem, SpaceEnvironments
import SplineProfile
from ObjectProperties import compute_sphere_properties, ObjectProperties, batch_sphere_properties

from models import MultiMagModel, SphereControlCommand
from solvers.modelOptimization import MultiMagSolver


def run_optimize(decisionVariable):
    solver = NewtonMethod(prob,**optimizerParams)
    result = solver.optimize(decisionVariable, max_iterations=30)
    return result


def getCost(decisionVariable):
    global prob
    return prob.cost(decisionVariable)


env = SpaceProblem(SpaceEnvironments.REGULATOR)
controller = PDController(env.start_state, 0.005, 0.00001, 0.005, 0.00001, 0.0, 0.000)
velocityProfile = SplineProfile.SplineProfile(env, env.times)


optimizer_type = GradientDescent

sigma_copper = 5.87e07
sigma_al = 3.77e07

expectation = False


# simulator object
if expectation:
    radii = np.random.normal(0.02,0.002,30)
    sigmas = np.random.normal(sigma_copper,1e7,30)
    object_properties = batch_sphere_properties(8940,radii,sigmas)

else:
    radius = 0.02
    mass, MOI = compute_sphere_properties(8940, radius)
    object_properties = ObjectProperties(
        mass=mass, MOI=MOI, radius=radius, sigma=sigma_copper
    )

model = MultiMagModel(env.magnets, object_properties, True)

results = []

if optimizer_type is NewtonMethod:
    max_iterations = 25
    optimizerParams = {
        "alpha": 1.0,
        "rho": 0.5,
        "c1": 1e-9,
        "FP_PRECISION": 1e-5,
        "min_alpha": 1e-15,
    }
else:
    max_iterations = 5000
    step = 1e3
    optimizerParams = {
        "alpha": step,
        "rho": 0.9,
        "c1": 1e-9,
        "FP_PRECISION":  1e-5*step,
        "min_alpha": step * 1e-25,
    }

def fixmaxDegree(angles):
    return (angles + 180) % 360 - 180


BRUTEFORCECOLOR = False
if __name__ == "__main__":
    builder = MultiMagSolver(
        model=model,
        state_cost_string=env.trajectory_dimensions,
        object_properties=object_properties,
        sphere_model=True,
        inverse_dynamics=True,
        num_processes=0,
        expectation=expectation,
    )

    for _ in range(100):
        np.random.seed(_)
        desired_wrench = 2 * (np.random.random((6, 1)) - 0.5) * 1e-3
        # desired_wrench[3:, 0] *= 1e-1
        desired_wrench[1, 0] *= -2

        builder.modify_problems(desired_wrench, np.zeros((3, 1)))
        prob = builder.problems[0]

        s = time.time()
        optimizer = optimizer_type(problem=prob, **optimizerParams)

        initial_positions = np.array([[20., 0, 0],
                                      [20., 90, 180]])
        optimization_results = [optimizer.optimize(
            initial_position.reshape((3, 1)), max_iterations=max_iterations) for initial_position in initial_positions]
        print("elapsed newton time ", time.time() - s)
        print('converged ', [result.converged for result in optimization_results])
        print('solutions ', [result.solution.flatten() for result in optimization_results])

        plt.figure()
        plt.plot(optimization_results[0].iterates[:,:,0])
        plt.figure()
        plt.plot(optimization_results[1].iterates[:,:,0])
        # print(solution,cost,converged,iterates.shape)

        strength = np.linspace(0, 40, 21)
        psis = np.linspace(-180, 180, 91)
        xis = np.linspace(-180, 180, 91)
        xx, yy, zz = np.meshgrid(strength, psis, xis)
        x = xx.reshape([-1])
        y = yy.reshape([-1])
        z = zz.reshape([-1])

        decisionVariablesList = []

        func = np.log

        for i in range(x.shape[0]):
            decisionVariables = np.array([x[i], y[i], z[i]]).reshape((3, 1))
            decisionVariablesList.append(decisionVariables)

        ctx = multiprocessing.get_context("spawn")

        costs = [
            getCost(decisionVariable) for decisionVariable in decisionVariablesList
        ]

        if BRUTEFORCECOLOR:
            with ctx.Pool(processes=8) as pool:
                results.extend(pool.map(run_optimize, decisionVariablesList))

            colors = []
            badSolutions = []
            for result in results:
                cost = result.cost
                solution = result.solution
                if cost < 1e-5:
                    colors.append("b")
                else:
                    colors.append("y")
                    badSolutions.append(solution)
        else:
            colors = ["b"] * x.shape[0]

        bestarg = np.argmin(costs)

        print(
            "bruteForce best ",
            x[bestarg],
            y[bestarg],
            z[bestarg],
            " with cost ",
            costs[bestarg])
        print('costs ', [result.cost for result in optimization_results])
        print('ratios ', [result.cost / costs[bestarg] for result in optimization_results])
        print("")
        print("")
        print(_)
        if min([result.cost / costs[bestarg] for result in optimization_results]) > 1 and x[bestarg] > 0.5:
            for params in [(90, 90), (45, 45), (45, 0), (0, 45), (90, 0), (0, 90), (180, 0), (0, 180)]:
                alt_result = optimizer.optimize(
                    np.array([20, params[0], params[1]]).reshape((3, 1)))
                print(params, alt_result.cost / costs[bestarg])
            break

        plot = False
        saveAnimation = False

        if plot:

            foundCosts = [[func(prob.cost(iterate)) for iterate in result.iterates] for result in optimization_results]

            npCosts = np.array(costs)
            npc = np.array(colors)

            for streng in strength:
                indices = np.argwhere(x == streng)[:, 0].tolist()
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(projection="3d")

                fig.suptitle(str(streng) + " dipole", fontsize=16)

                ax.scatter(
                    y[indices], z[indices], func(npCosts[indices]), s=1, c=npc[indices]
                )

                if saveAnimation:

                    def animate(i):
                        ax.view_init(elev=10.0, azim=i)
                        return (fig,)


                    # Animate
                    anim = animation.FuncAnimation(
                        fig, animate, frames=360, interval=20, blit=True
                    )
                    # Save
                    anim.save("basic_animation.mp4", fps=30)
                else:
                    for result, found_costs in zip(optimization_results, foundCosts):
                        ax.plot(
                            fixmaxDegree(result.iterates[:, 1, 0]),
                            fixmaxDegree(result.iterates[:, 2, 0]),
                            found_costs,
                            c="r",
                            markersize=200,
                            linewidth=6,
                        )
            plt.show()
            break
