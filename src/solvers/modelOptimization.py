#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:28:31 2021

@author: tabor
"""
import numpy as np
import torch
import time
from ll4ma_opt.problems import Problem

import matplotlib.pyplot as plt

import torch.multiprocessing as multiprocessing

from ll4ma_opt.solvers.line_search import (
    NewtonMethod,
    GaussNewtonMethod,
    GradientDescent,
)

from models import (
    rot_z,
    model_call_rotated_wrench,
    get_omni_frame,
    CylinderControlCommand,
    SphereControlCommand,
)
from ObjectProperties import build_mass_matrix

torch.set_default_tensor_type(torch.DoubleTensor)

sphere_parameters = {
    "alpha": 1.0,
    "rho": 0.5,
    "c1": 1e-9,
    "FP_PRECISION": 1e-5,
    "min_alpha": 1e-10,
}
cylinder_parameters = {
    "alpha": 1.0,
    "rho": 0.5,
    "c1": 1e-9,
    "FP_PRECISION": 1e-15,
    "min_alpha": 1e-15,
}


def sphere_optimize(prob):
    try:
        solver = NewtonMethod(problem=prob, **sphere_parameters)
        result = solver.optimize(prob.initial_solution, max_iterations=25)
        return result
    except:
        return False


def cylinder_optimize(prob):
    solver = NewtonMethod(problem=prob, **cylinder_parameters)
    result = solver.optimize(prob.initial_solution, max_iterations=25)
    return result


def build_acceleration_q(state_cost_string):
    return build_q(state_cost_string, 2e4, 1)


def build_force_q(state_cost_string):
    return build_q(state_cost_string)


def build_q(state_cost_string, linear_cost=1.0, orientation_cost=200):
    letter_options = ["x", "y", "z", "r", "p", "t"]
    activated = [0] * 6
    if "all" in state_cost_string:
        state_cost_string = letter_options
    for index in range(len(letter_options)):
        if letter_options[index] in state_cost_string:
            if index < 3:
                activated[
                    index
                ] = linear_cost  # / (objectProperties[0] * tolerances[0])
            else:
                activated[
                    index
                ] = orientation_cost  # / (objectProperties[1][0,0] * tolerances[1])

    Q = np.diag(activated)
    return Q


class CylinderOptimization(Problem):
    def __init__(self, weights):
        super().__init__()
        self.initial_solution = np.array([20, 0]).reshape((2, 1))
        self.max_bounds[0] = 40
        self.min_bounds[0] = 0.1

        self.desired = torch.zeros((6, 1))
        self.unit_strength_wrench = torch.zeros((6, 1))
        self.frame = torch.eye(3)
        self.rho_mode = False
        self.weights = torch.tensor(weights)

    def size(self):
        return 2

    def pointing_model_loss(self, x, frame):
        selected_wrench = x[0] ** 2 * torch.cat(
            (
                frame @ self.unit_strength_wrench[:3],
                frame @ self.unit_strength_wrench[3:],
            )
        )
        error = 1e13 * (self.desired - selected_wrench)
        cost = error.t() @ self.weights @ error
        return cost

    def rho_loss(self, x):
        frame = torch.mm(self.frame, rot_z(x[1], torch))
        return self.pointing_model_loss(x, frame)

    def tensor_cost(self, x):
        if self.rho_mode:
            return self.rho_loss(x)
        else:
            return self.pointing_model_loss(x, self.frame)

    def visualize_optimization(self, iterates):
        plt.plot(iterates[:, 0, 0], iterates[:, 1, 0])


class SphericalOptimization(Problem):
    def __init__(self, weights):
        super().__init__()
        self.initial_solution = np.array([20, 0, 0]).reshape((3, 1))
        self.min_bounds[0] = 0.1
        self.max_bounds[0] = 40

        self.desired = torch.zeros((6, 1))

        self.weights = torch.tensor(weights)

        self.frho0 = torch.tensor(0.0)
        self.trho0 = torch.tensor(0.0)
        self.frho90 = torch.tensor(0.0)
        self.fphi90 = torch.tensor(0.0)
        self.ttheta90 = torch.tensor(0.0)

        self.rho = torch.tensor(np.zeros((3, 1)))

        self.zero = torch.zeros(1)

    def size(self):
        return 3

    def get_wrench(self, x):
        return model_call_rotated_wrench(
            x[0],
            x[1:],
            (self.frho0, self.trho0, self.frho90, self.fphi90, self.ttheta90),
            self.rho,
            library=torch,
            zero=self.zero,
        )

    def tensor_cost(self, x):
        selected_wrench = self.get_wrench(x)
        error = 1e10 * (self.desired - selected_wrench)[0]
        cost = error.t() @ self.weights @ error
        return cost

    def tensor_error(self, x):
        selected_wrench = self.get_wrench(x)
        error = 1e5 * (self.desired - selected_wrench)[0]
        weighted_error = torch.sqrt(self.weights) @ error
        return weighted_error

    def visualize_optimization(self, iterates):
        plt.plot(iterates[:, 0, 0], iterates[:, 1, 0])


class SphericalAccelOptimization(SphericalOptimization):
    def __init__(self, weights):
        super().__init__(weights)
        self.mass_matrix_inv = torch.eye(6)

    def tensor_cost(self, x):
        selected_wrench = self.get_wrench(x)
        error = 1e12 * (self.desired - self.mass_matrix_inv @ selected_wrench)[0]
        cost = error.t() @ self.weights @ error
        return cost

    def tensor_error(self, x):
        selected_wrench = self.get_wrench(x)
        error = 1e5 * (self.desired - self.mass_matrix_inv @ selected_wrench)[0]
        weighted_error = torch.sqrt(self.weights) @ error
        return weighted_error


def get_cost(iterate):
    return iterate["problem"].cost(iterate["variables"])


class MultiMagSolver:
    def __init__(
        self,
        model,
        state_cost_string,
        object_properties,
        sphere_model=False,
        inverse_dynamics=False,
        boat=None,
        num_processes=8,
    ):
        self.model = model
        self.problems = []
        self.time = 0
        self.use_sphere_model = sphere_model
        self.boat = boat
        self.inverse_dynamics = inverse_dynamics
        self.object_properties = object_properties

        self.costs = []
        self.selected_optimizer = NewtonMethod
        self.cost_time = 0
        self.gradient_time = 0
        self.hessian_time = 0

        if inverse_dynamics:
            qweights = build_acceleration_q(state_cost_string)
        else:
            qweights = build_force_q(state_cost_string)
        print("controller weights ", qweights)

        if sphere_model:
            for i in range(2 * len(model.magnets)):
                if inverse_dynamics:
                    prob = SphericalAccelOptimization(qweights)
                else:
                    prob = SphericalOptimization(qweights)
                if i % 2 == 0:
                    prob.initial_solution = np.array([20, 0, 0]).reshape((3, 1))
                if i % 2 == 1:
                    prob.initial_solution = np.array([20, 180, 180]).reshape((3, 1))
                self.problems.append(prob)
        else:
            for i in range(3 * len(model.magnets)):
                prob = CylinderOptimization(qweights)
                prob.rho_mode = (i % 3) == 2
                self.problems.append(prob)
        ctx = multiprocessing.get_context("spawn")
        if len(self.problems) > num_processes:
            num_processes = len(self.problems)
        self.pool = ctx.Pool(processes=num_processes)

        self.iterations = []
        self.times = []

    def modify_problems(self, desired_wrench, position):
        if self.use_sphere_model:
            self.modify_sphere_problems(desired_wrench, position)
        else:
            self.modify_cylinder_problems(desired_wrench, position)

    def modify_sphere_problems(self, desired, position):
        relative_poses = np.array(self.model.magnets) - np.transpose(position)

        if self.inverse_dynamics:
            object_properties = self.object_properties
            if self.boat:
                object_properties = object_properties + self.boat
            mass_matrix = torch.tensor(build_mass_matrix(object_properties, np, False))
            mass_matrix_inv = torch.linalg.inv(mass_matrix)

        for i in range(len(self.model.magnets)):
            chosen_magnet_pose = relative_poses[i]
            rho = (-1 * chosen_magnet_pose).reshape((3, 1))
            dist = np.linalg.norm(chosen_magnet_pose)
            frho0, trho0, frho90, fphi90, ttheta90 = self.model.sphere_model.model_call(
                dist, 1
            )
            for j in range(2):
                self.problems[i * 2 + j].frho0 = torch.tensor(frho0)
                self.problems[i * 2 + j].trho0 = torch.tensor(trho0)
                self.problems[i * 2 + j].frho90 = torch.tensor(frho90)
                self.problems[i * 2 + j].fphi90 = torch.tensor(fphi90)
                self.problems[i * 2 + j].ttheta90 = torch.tensor(ttheta90)

                self.problems[i * 2 + j].desired = torch.Tensor(desired)
                self.problems[i * 2 + j].rho = torch.tensor(rho)

                if self.inverse_dynamics:
                    self.problems[i * 2 + j].mass_matrix_inv = mass_matrix_inv

    def modify_cylinder_problems(self, desired, position):
        relative_poses = (
            np.array(self.model.magnets) - np.transpose(position)
        ).tolist()

        for i in range(len(self.model.magnets)):
            chosen_magnet_pose = relative_poses[i]
            omni_frame = get_omni_frame(chosen_magnet_pose)
            dist = np.linalg.norm(chosen_magnet_pose)
            wrenches = self.model.cylinder_model.model_call(
                dist, 1
            )  # 1 dipole strength
            for j in range(3):
                self.problems[i * 3 + j].frame = torch.Tensor(omni_frame)
                self.problems[i * 3 + j].desired = torch.Tensor(desired)
                self.problems[i * 3 + j].unit_strength_wrench = torch.tensor(
                    wrenches[j]
                )

    def solve_wrench_cylinder(self, position, desired_wrench):
        s = time.time()
        self.modify_problems(desired_wrench, position)
        results = self.pool.map(cylinder_optimize, self.problems)
        iterations = [result.iterates.shape[0] for result in results]
        self.iterations.append((iterations))
        no_action_cost = self.problems[0].cost(np.zeros((2, 1)))
        costs = [result.cost for result in results]
        self.costs.append(costs)
        best_action = np.argmin(costs)

        if costs[best_action] > no_action_cost:
            control = CylinderControlCommand()
            # print('shit')
        else:
            mag_num = best_action // 3
            model_num = best_action % 3
            strength = results[best_action].solution[0, 0]
            theta = results[best_action].solution[1, 0]
            control = CylinderControlCommand(
                mag_num=mag_num,
                model_num=model_num,
                dipole_strength=strength,
                theta=theta,
            )

        timestep = time.time() - s
        self.times.append(timestep)
        self.time += timestep
        return control

    def solve_wrench(self, position, desired_wrench):
        if self.use_sphere_model:
            return self.solve_wrench_sphere(position, desired_wrench)
        else:
            return self.solve_wrench_cylinder(position, desired_wrench)

    def solve_wrench_sphere(self, position, desired_wrench):

        s = time.time()
        self.modify_problems(desired_wrench, position)
        # results = [sphere_optimize(prob) for prob in self.problems]
        results = self.pool.map(sphere_optimize, self.problems)

        iterations = [result.iterates.shape[0] for result in results]
        # self.cost_time += np.sum([result[4] for result in results])
        # self.gradient_time += np.sum([result[5] for result in results])
        # self.hessian_time += np.sum([result[6] for result in results])

        self.iterations.append((iterations))
        no_action_cost = self.problems[0].cost(np.zeros((3, 1)))
        costs = [result.cost for result in results]
        self.costs.append(costs)

        best_action = np.argmin(costs)

        if costs[best_action] > no_action_cost:
            control = SphereControlCommand()
            # print('shit')
        else:
            mag_num = best_action // 2
            strength = results[best_action].solution[0, 0]
            psi = results[best_action].solution[1, 0]
            xi = results[best_action].solution[2, 0]
            control = SphereControlCommand(
                mag_num=mag_num, dipole_strength=strength, psi=psi, xi=xi
            )

        timestep = time.time() - s
        self.times.append(timestep)
        self.time += timestep
        return control

    def brute_sphere_solver(self, position, desired_wrench):
        self.modify_problems(desired_wrench, position)

        s = time.time()
        iterates = []
        for mag in range(len(self.model.magnets)):
            for i in range(-180, 180, 3):
                for j in range(-180, 180, 3):
                    for k in range(0, 40):

                        iterate = {
                            "problem": self.problems[mag * 2],
                            "variables": np.array((k, i, j)).reshape((3, 1)),
                            "mag": mag,
                        }
                        iterates.append(iterate)

        results = self.pool.map(get_cost, iterates)
        index = np.argmin(results)
        print("brute time ", time.time() - s)
        selected = iterates[index]

        variables = selected["variables"]
        control = SphereControlCommand(
            mag_num=selected["mag"],
            dipole_strength=variables[0, 0],
            psi=variables[1, 0],
            xi=variables[2, 0],
        )
        return control
