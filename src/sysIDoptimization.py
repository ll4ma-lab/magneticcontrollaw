#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 20:30:42 2021

@author: tabor
"""

import numpy as np
import time
import torch
from models import model_call_rotated_wrench, SphereModel
from ObjectProperties import ObjectProperties, build_mass_matrix, compute_sphere_properties

from solvers.modelOptimization import build_q, build_acceleration_q
from ll4ma_opt.problems import Problem
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ll4ma_opt.solvers.line_search import NewtonMethod


class RadiusSysid(Problem):
    def __init__(self, weights, object_properties=None, boat=None):
        super().__init__()

        self.single_model = SphereModel(1)
        self.sigma_multiplier = 1e7

        self.error_multiplier = 1e6

        if object_properties:
            self.initial_solution = np.array(
                [
                    object_properties.radius,
                    object_properties.sigma / self.sigma_multiplier,
                ]
            ).reshape((-1, 1))
        else:
            self.initial_solution = np.ones((2, 1)) * 0.001
            self.initial_solution[1] = 5.8e7 / self.sigma_multiplier

        self.min_bounds[0] = 1e-3
        self.max_bounds[0] = 5e-2

        self.min_bounds[1] = 2e5 / self.sigma_multiplier
        self.max_bounds[1] = 2e8 / self.sigma_multiplier

        self.weights = self.make_tensor(weights)
        if boat:
            self.boat = boat.torch()
        else:
            self.boat = boat

    def update_vals(
        self,
        controls,
        magnets,
        positions,
        linear_velocity,
        rotational_velocity,
        drag_terms=(0, 0),
    ):
        # remove first control and position as we dont know its acceleration

        mag_nums = [control.mag_num for control in controls[1:]]
        chosen_magnet_pose = np.array(magnets)[mag_nums] - positions[1:]
        dipole_strength = [control.dipole_strength for control in controls[1:]]
        angles = np.array([(control.psi, control.xi) for control in controls[1:]]).T
        timesteps = np.array([control.seconds for control in controls[1:]]).reshape(
            (-1, 1)
        )

        self.n = len(controls[1:])

        rhos = -chosen_magnet_pose.T
        dists = np.linalg.norm(chosen_magnet_pose, axis=1)
        zero = np.array([0] * self.n)

        self.dipole_strength = self.make_tensor(dipole_strength)
        self.angles = self.make_tensor(angles)
        self.rhos = self.make_tensor(rhos)
        self.dists = self.make_tensor(dists)

        self.zero = self.make_tensor(zero)

        delta_v = np.hstack(
            (
                linear_velocity[1:, :] - linear_velocity[:-1, :],
                rotational_velocity[1:, :] - rotational_velocity[:-1, :],
            )
        )
        np_a = delta_v / timesteps

        drag_force = (
            drag_terms[0] * linear_velocity[:-1, :]
            + np.sign(linear_velocity[:-1, :])
            * drag_terms[1]
            * linear_velocity[:-1, :] ** 2
        )

        self.drag_wrenches = self.make_tensor(
            np.hstack((-drag_force, np.zeros_like(drag_force)))
        )
        self.accelerations = self.make_tensor(np_a.T)

        self.material_density = self.make_tensor([8940.0])[0]

    def forward_acceleration(
        self, object_properties, dists, dipole_strengths, angles, rhos, zero
    ):

        mass_matrix = self.construct_mass_matrix(object_properties, torch)
        model_vals = self.single_model.model_call(
            dists, 1, radii=object_properties.radius, sigma=object_properties.sigma
        )
        model_vals = [model_val.squeeze() for model_val in model_vals]
        modeled_wrenches = model_call_rotated_wrench(
            dipole_strengths, angles, model_vals, rhos, library=torch, zero=zero
        ).squeeze()

        # print(modeled_wrenches)
        # print(self.drag_wrenches)
        wrenches = modeled_wrenches + self.drag_wrenches

        return torch.linalg.inv(mass_matrix) @ wrenches.T

    def get_object_properties(self, x, library=np):
        radius = x[0, 0]
        sigma = x[1, 0]
        mass, MOI = compute_sphere_properties(self.material_density, radius, library)
        object_properties = ObjectProperties(
            mass=mass, MOI=MOI, radius=radius, sigma=sigma * self.sigma_multiplier
        )
        return object_properties

    def construct_mass_matrix(self, object_properties, library=np):
        if self.boat:
            object_properties = object_properties + self.boat
        M = build_mass_matrix(object_properties, library, False)
        return M

    def size(self):
        return 2

    def tensor_cost(self, x):
        object_properties = self.get_object_properties(x, torch)

        modeled_accelerations = self.forward_acceleration(
            object_properties,
            self.dists,
            self.dipole_strength,
            self.angles,
            self.rhos,
            self.zero,
        )
        error = (self.accelerations - modeled_accelerations) * self.error_multiplier

        left = (error.T @ self.weights).unsqueeze(1)
        right = error.T.unsqueeze(2)
        cost = (
            torch.sum(torch.bmm(left, right)) / self.rhos.shape[0]
        )  # -  5e14* (x[0:1])**2

        return cost

    def tensor_error(self, x):
        object_properties = self.get_object_properties(x, torch)

        modeled_accelerations = self.forward_acceleration(
            object_properties,
            self.dists,
            self.dipole_strength,
            self.angles,
            self.rhos,
            self.zero,
        )
        error = (self.accelerations - modeled_accelerations) * self.error_multiplier

        weighted_error = error.T @ torch.sqrt(self.weights / self.rhos.shape[0])
        return weighted_error.reshape(((-1, 1)))

    def visualize_optimization(
        self, iterates, samples=50, func=np.abs, show_costs=False
    ):
        fig = plt.figure(figsize=(10, 10))
        if show_costs:
            ax = fig.add_subplot(projection="3d")
            start_angle = -100
            ax.view_init(elev=17.0, azim=start_angle)
        else:
            ax = fig.add_subplot()

        possible_xs = np.linspace(self.min_bounds[0], self.max_bounds[0], samples)
        possible_ys = np.linspace(self.min_bounds[1], self.max_bounds[1], samples)
        xs, ys = np.meshgrid(possible_xs, possible_ys)
        xs = xs.flatten()
        ys = ys.flatten()
        possible_costs = [
            func(self.cost(np.stack((xs[i], ys[i])).reshape((2, 1))))
            for i in range(len(xs))
        ]
        if func is np.log:
            clip_value = np.median(possible_costs)
        else:
            clip_value = 2 * np.median(possible_costs)
        possible_costs = np.array(possible_costs)
        possible_costs[possible_costs > clip_value] = clip_value

        best_index = np.argmin(possible_costs)
        print(
            "best brute force  func cost ",
            possible_costs[best_index],
            " at ",
            xs[best_index],
            ys[best_index],
        )
        print(
            "result func cost ",
            min(
                [
                    min(func(self.cost(point.reshape((-1, 1)))), clip_value)
                    for point in iterates[-1].reshape((-1, 2))
                ]
            ),
        )

        ax.set_xlim(self.min_bounds[0], self.max_bounds[0])
        ax.set_ylim(self.min_bounds[1], self.max_bounds[1])

        points = iterates[0].reshape((-1, 2))
        costs = [
            [min(func(self.cost(point.reshape((-1, 1)))), clip_value)]
            for point in points
        ]
        if show_costs:
            ax.scatter(xs, ys, possible_costs, color=(0, 0, 1, 0.5))
            # ax.set_zlim(0, clip_value)
            points_3d = np.concatenate((points, np.array(costs)), 1)
            # silly hack to get filled circle
            scat = ax.scatter(
                points_3d[:, 0],
                points_3d[:, 1],
                points_3d[:, 2],
                linewidths=10,
                c="orange",
            )
            scat2 = ax.scatter(
                points_3d[:, 0],
                points_3d[:, 1],
                points_3d[:, 2],
                linewidths=10,
                c="orange",
            )

            # scat._offsets3d = (points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
        else:
            scat = plt.scatter(points[:, 0], points[:, 1], linewidths=10, c="orange")
            scat.set_offsets(points)
            scat2 = None

        def update(num, scat, scat2, iterates, show_costs, ax):

            ax.set_title(num)
            if show_costs:
                ax.view_init(elev=17.0, azim=start_angle + num / 3)
            points = iterates[num].reshape((-1, 2))
            costs = [
                [min(func(self.cost(point.reshape((-1, 1)))), clip_value)]
                for point in points
            ]
            points_3d = np.concatenate((points, np.array(costs)), 1)
            # print(points_3d.shape)
            if show_costs:
                scat._offsets3d = (points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
                scat2._offsets3d = (points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])

            else:
                scat.set_offsets(points)

        anim = FuncAnimation(
            fig,
            update,
            fargs=(scat, scat2, iterates, show_costs, ax),
            frames=len(iterates),
            interval=100,
            repeat=True,
        )
        return anim

    def visualize_predicted_acceleration(self, iterate):
        fig, ax = plt.subplots(3)
        object_properties = self.get_object_properties(self.make_tensor(iterate), torch)
        modeled_accelerations = self.forward_acceleration(
            object_properties,
            self.dists,
            self.dipole_strength,
            self.angles,
            self.rhos,
            self.zero,
        )
        fig.suptitle("Modeled vs Measured Accelerations")

        titles = ["x", "y", "θ"]

        for fig_index, acceleration_index in enumerate([0, 1, 5]):
            ax[fig_index].set_title(titles[fig_index])
            ax[fig_index].plot(self.accelerations[acceleration_index], label="measured")
            ax[fig_index].plot(
                modeled_accelerations[acceleration_index], label="predicted"
            )
            ax[fig_index].legend()


class SeperateRadiusSysid(RadiusSysid):
    def __init__(self, weights, object_properties, boat=None):
        super().__init__(weights=weights, boat=boat)

        if object_properties:
            self.initial_solution = np.array(
                [
                    object_properties.radius,
                    object_properties.sigma / self.sigma_multiplier,
                    object_properties.radius,
                ]
            ).reshape((-1, 1))
        else:
            self.initial_solution[2] = 0.1

        # radius and sigma keep the same

        # second radius for mass and MOI
        self.min_bounds[2] = 1e-5
        self.max_bounds[2] = 1e-0

    def size(self):
        return 3

    def get_object_properties(self, x, library=np):
        radius = x[0, 0]
        sigma = x[1, 0]
        mass_radius = x[2, 0]
        mass, MOI = compute_sphere_properties(self.material_density, mass_radius, library)
        object_properties = ObjectProperties(
            mass=mass, MOI=MOI, radius=radius, sigma=sigma * self.sigma_multiplier
        )
        return object_properties

    def visualize_optimization(self, iterates, func=np.abs):
        fig, axs = plt.subplots(self.size(), 1)

        for i in range(self.size()):
            axs[i].plot(iterates[:, i, 0], "r")
        fig.suptitle("Iterates over time")

        fig, axs = plt.subplots(self.size(), 1)
        fig.suptitle("Cost as iterate changes")

        for i in range(self.size()):
            min_value = self.min_bounds[i]
            max_value = self.max_bounds[i]
            xs = np.linspace(min_value, max_value, 100)

            ys = []
            for x in xs:
                iterate = np.copy(iterates[-1])
                iterate[i] = x
                ys.append(self.cost(iterate))

            x = iterates[-1][i]
            y = self.cost(iterates[-1])
            axs[i].plot([x], func([y]), "r", marker="o")
            axs[i].plot(xs, func(ys), "b")


class MassSysID(SeperateRadiusSysid):
    def __init__(self, weights, object_properties=None, boat=None):
        super().__init__(
            weights=weights, object_properties=object_properties, boat=boat
        )

        if object_properties:
            self.initial_solution = np.array(
                [
                    object_properties.radius,
                    object_properties.sigma / self.sigma_multiplier,
                    object_properties.mass,
                    object_properties.MOI[0, 0],
                    object_properties.MOI[1, 1],
                    object_properties.MOI[2, 2],
                ]
            ).reshape((-1, 1))
        else:
            self.initial_solution = np.array(
                [
                    self.initial_solution[0, 0],
                    self.initial_solution[1, 0],
                    1.0,  # mass
                    1.0,  # Ixx
                    1.0,  # Iyy
                    1.0,  # Izz
                ]
            ).reshape((-1, 1))

        self.min_bounds[0] = 1e-3

        self.min_bounds[1] = 1e4 / self.sigma_multiplier

        # radius and sigma keep the same
        self.min_bounds[2] = 1e-10
        self.max_bounds[2] = 2

        self.min_bounds[3] = 1e-5  # mass
        self.min_bounds[3:] = 1e-10  # MOI

        self.max_bounds[3] = 2  # mass
        self.max_bounds[3:] = 1e0  # MOI

    def size(self):
        return 6

    def get_object_properties(self, x, library=np):
        radius = x[0, 0]
        sigma = x[1, 0]
        mass = x[2, 0]
        MOI = library.diag(x[3:, 0])
        object_properties = ObjectProperties(
            mass=mass, MOI=MOI, radius=radius, sigma=sigma * self.sigma_multiplier
        )

        return object_properties


def sys_id_demo(
    controls,
    magnets,
    positions,
    linear_velocity,
    rotational_velocity,
    sys_id_type=RadiusSysid,
    boat=None,
):

    optimizer_params = {
        "alpha": 1,
        "rho": 0.9,
        "c1": 1e-9,
        "FP_PRECISION": 1e-8,
        "min_alpha": 1e-20,
    }
    prob = sys_id_type(build_acceleration_q("all"), boat)  # built with M(0.04) Q M(0.04)
    optimizer = NewtonMethod(problem=prob, **optimizer_params)
    prob.updateVals(controls, magnets, positions, linear_velocity, rotational_velocity)
    result = optimizer.optimize(prob.initial_solution, max_iterations=50)
    prob.visualize_optimization(result.iterates)
    return result
