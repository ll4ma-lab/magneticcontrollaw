#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 13:02:06 2021

@author: tabor
"""
import sysIDoptimization
from ll4ma_opt.solvers import Momentum, NewtonMethod
from solvers.modelOptimization import build_acceleration_q
import numpy as np
import matplotlib.pyplot as plt
from Stein import SteinSysid

class Adaptive:
    def __init__(
        self,
        magnets,
        dimensions,
        object_properties,
        batch_size=1,
        internal_sys_id=sysIDoptimization.RadiusSysid,
        boat=False,
        drag_terms=(0, 0, 0, 0),
        stein=False
    ):

        self.magnets = magnets
        self.object_properties = object_properties
        Q = build_acceleration_q(dimensions)
        print("adaptive weights ", Q)
        prob = internal_sys_id(
            Q, object_properties, boat=boat, drag_terms=drag_terms
        )
        if stein:
            self.problem = SteinSysid(prob,30,1e3)
        else:
            self.problem = prob
        self.current_solution = self.problem.initial_solution
        optimizer_params = {"alpha": 1, "rho": 0.9, "min_alpha": 1e-20, "c1": 1e-30}

        step = 1e-9  # / batch_size

        optimizer_params = {
            "alpha": step,
            "rho": 0.5,
            "c1": 1e-9,
            "FP_PRECISION": 1e-15,
            "min_alpha": 1e-20,
            "momentum": 0.9,
        }

        self.optimizer = Momentum(problem=self.problem, **optimizer_params)
        self.batch_size = batch_size

        self.linear_velocities = []
        self.rotational_velocities = []
        self.controls = []
        self.positions = []
        self.parameters = []

    def reset_current_solution(self, object_properties):
        self.current_solution = self.problem.reset_current_solution(object_properties)

    def add_state(self, state):
        current_linear_velocity = state[0][:3, :3] @ state[2][:, 0]
        current_rotational_velocity = state[0][:3, :3] @ state[1][:, 0]

        self.linear_velocities.append(current_linear_velocity)
        self.rotational_velocities.append(current_rotational_velocity)
        self.positions.append(np.copy(state[0][:3, 3]))

    def step(self, state, control):
        self.add_state(state)
        self.controls.append(control)

        if len(self.parameters) == 0:
            self.parameters.append(self.current_solution)

        batch_size = min(self.batch_size, len(self.controls) - 1)
        if len(self.controls) - 1 < self.batch_size + 5:
            return

        selected_controls = self.controls[-batch_size:]
        selected_positions = np.array(self.positions[-batch_size:])
        selected_linear_velocities = np.array(self.linear_velocities[-batch_size:])
        selected_rotation_velocities = np.array(
            self.rotational_velocities[-batch_size:]
        )

        self.problem.update_vals(
            selected_controls,
            self.magnets,
            selected_positions,
            selected_linear_velocities,
            selected_rotation_velocities,
        )

        result = self.optimizer.optimize(self.current_solution, max_iterations=1)
        self.current_solution = result.solution
        self.parameters.append(self.current_solution)

        updated_properties = self.problem.get_object_properties(result.solution[:2], np)
        self.object_properties.radius = updated_properties.radius
        self.object_properties.sigma = updated_properties.sigma
        self.object_properties.mass = updated_properties.mass
        self.object_properties.MOI = updated_properties.MOI

    def plot(self, true_vector):
        parameters = np.stack(self.parameters)
        fig, axs = plt.subplots(self.problem.size(), 1)

        for i in range(self.problem.size()):
            axs[i].plot(parameters[:, i], "r")
            axs[i].plot([0, parameters.shape[0]], [true_vector[i], true_vector[i]])
