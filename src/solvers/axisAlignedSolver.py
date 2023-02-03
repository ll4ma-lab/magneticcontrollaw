#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 12:32:40 2022

@author: tabor
"""
from ll4ma_opt.solvers.line_search import NewtonMethod

from models import CylinderControlCommand
from solvers.modelOptimization import build_q
from ll4ma_opt.problems import Problem
import torch
import numpy as np
import time

torch.set_default_tensor_type(torch.DoubleTensor)


class AxisAlignedProblem(Problem):
    def __init__(self, A, desired_wrench, weights):
        self.num_wrenches = A.shape[1]
        super().__init__()

        self.initial_solution = np.ones((self.num_wrenches, 1)) * 1e-10
        self.A = torch.Tensor(A)
        self.desired_wrench = torch.Tensor(desired_wrench)
        self.weights = torch.Tensor(weights)

        self.min_bounds[:] = 0
        self.max_bounds[:] = 40
        self.wrench_multiplier = (
            1  # self.num_wrenches/2 #assume half of the control options are useful
        )

    def size(self):
        return self.num_wrenches

    def tensor_cost(self, x):
        selected_wrench = self.A @ (x * x) / self.wrench_multiplier
        error = 1e12 * (self.desired_wrench - selected_wrench)
        cost = error.t() @ self.weights @ error
        return cost


class AxisAlignedSolver:
    def __init__(self, model, state_cost_string):
        self.Q = build_q(state_cost_string)
        self.model = model
        self.count = 0
        self.time = 0

    def solve_wrench(self, position, desired_wrench):
        start = time.time()
        model_id = self.count % (6 * len(self.model.magnets))
        self.count += 1
        if model_id == 0:
            self.controls_possible = []
            for i in range(len(self.model.magnets)):
                self.controls_possible.append(
                    CylinderControlCommand(mag_num=i, model_num=0, dipole_strength=1)
                )
            for i in range(len(self.model.magnets)):
                self.controls_possible.append(
                    CylinderControlCommand(mag_num=i, model_num=1, dipole_strength=1)
                )
            for theta in [0, 90, 180, -90]:
                for i in range(len(self.model.magnets)):

                    self.controls_possible.append(
                        CylinderControlCommand(
                            mag_num=i, model_num=2, dipole_strength=1, theta=theta
                        )
                    )
            wrenches_possible = [
                self.model.forwardCylinder(position, control)
                for control in self.controls_possible
            ]
            A = np.array(wrenches_possible).squeeze(2).T
            self.A = A
            problem = AxisAlignedProblem(A, desired_wrench, self.Q)
            solver = NewtonMethod(problem)
            result = solver.optimize(problem.initial_solution)
            print(result.iterates.shape, result.converged, result.cost)
            self.dipole_strengths = result.solution

        control = self.controls_possible[model_id]
        control.dipole_strength = self.dipole_strengths[model_id, 0]
        self.time += time.time() - start
        return control
