#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 20:30:42 2021

@author: tabor
"""

import numpy as np
import time
import torch
from tqdm import tqdm
from models import model_call_rotated_wrench, SphereModel
from ObjectProperties import (
    ObjectProperties,
    build_mass_matrix,
    compute_sphere_properties,
)
from matplotlib.patches import Ellipse
from decimal import Decimal

from solvers.modelOptimization import build_q, build_acceleration_q
from ll4ma_opt.problems import Problem
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
from ll4ma_opt.solvers.line_search import NewtonMethod
from visualization import Lanc1,Lanc2,Lanc3

class RadiusSysid(Problem):
    def __init__(
        self, weights, object_properties=None, boat=None, drag_terms=(0, 0, 0, 0)
    ):
        super().__init__()

        self.single_model = SphereModel(1)
        self.sigma_multiplier = 1e9

        self.error_multiplier = 1e6

        self.regularization_multiplier = 1e10
        self.reset_current_solution(object_properties)

        self.min_bounds[0] = 1e-3
        self.max_bounds[0] = 1e-1

        self.min_bounds[1] = 2e6 / self.sigma_multiplier
        self.max_bounds[1] = 8e7 / self.sigma_multiplier

        self.prior = self.make_tensor((self.max_bounds - self.min_bounds) / 2)

        self.weights = self.make_tensor(weights)
        if boat:
            self.boat = boat.torch()
        else:
            self.boat = boat

        self.param_names = [
            "Radius (m)",
            "Conductivity (S/m) x 10^7",
        ]

        self.drag_terms = drag_terms

        # self.make_tensor = lambda x:torch.Tensor(x).cuda()

    def reset_current_solution(self, object_properties=None):
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
        return self.initial_solution

    def update_vals(
        self,
        controls,
        magnets,
        positions,
        linear_velocity,
        rotational_velocity,
    ):
        velocity_shift = 0

        # remove last control and position as we dont know its acceleration

        mag_nums = [control.mag_num for control in controls[: -1 - velocity_shift]]
        chosen_magnet_pose = (
            np.array(magnets)[mag_nums] - positions[: -1 - velocity_shift]
        )
        dipole_strength = [
            control.dipole_strength for control in controls[: -1 - velocity_shift]
        ]
        angles = np.array(
            [(control.psi, control.xi) for control in controls[: -1 - velocity_shift]]
        ).T
        timesteps = np.array(
            [control.seconds for control in controls[: -1 - velocity_shift]]
        ).reshape((-1, 1))

        self.linear_velocity = self.make_tensor(
            linear_velocity[1 + velocity_shift :, :]
        )
        self.rotational_velocity = self.make_tensor(
            rotational_velocity[1 + velocity_shift :, :]
        )

        self.n = len(dipole_strength)

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
                linear_velocity[1 + velocity_shift :, :]
                - linear_velocity[velocity_shift:-1, :],
                rotational_velocity[1 + velocity_shift :, :]
                - rotational_velocity[velocity_shift:-1, :],
            )
        )
        np_a = delta_v / timesteps

        drag_force = (
            self.drag_terms[0] * linear_velocity[velocity_shift:-1, :]
            + np.sign(linear_velocity[velocity_shift:-1, :])
            * self.drag_terms[1]
            * linear_velocity[velocity_shift:-1, :] ** 2
        )

        drag_torque = (
            self.drag_terms[2] * rotational_velocity[velocity_shift:-1, :]
            + np.sign(rotational_velocity[velocity_shift:-1, :])
            * self.drag_terms[3]
            * rotational_velocity[velocity_shift:-1, :] ** 2
        )

        self.drag_wrenches = self.make_tensor(np.hstack((-drag_force, -drag_torque)))
        self.accelerations = self.make_tensor(np_a.T)

        self.material_density = self.make_tensor([8940.0])[0]

    def forward_acceleration(
        self, x, object_properties, dists, dipole_strengths, angles, rhos, zero
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
        wrenches = modeled_wrenches + self.get_drag_wrenches(x)

        return torch.linalg.inv(mass_matrix) @ wrenches.T

    def get_drag_wrenches(self, x):
        return self.drag_wrenches

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
        object_properties = self.get_object_properties(x, library=torch)

        modeled_accelerations = self.forward_acceleration(
            x,
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
            torch.sum(torch.bmm(left, right)) / self.rhos.shape[1]
        )  # -  5e14* (x[0:1])**2

        prior_error = x - self.prior
        cost = cost + prior_error.T @ prior_error * self.regularization_multiplier
        return cost

    def tensor_error(self, x):
        object_properties = self.get_object_properties(x, library=torch)

        modeled_accelerations = self.forward_acceleration(
            x,
            object_properties,
            self.dists,
            self.dipole_strength,
            self.angles,
            self.rhos,
            self.zero,
        )
        error = (self.accelerations - modeled_accelerations) * self.error_multiplier

        weighted_error = error.T @ torch.sqrt(self.weights / self.rhos.shape[1])
        return weighted_error.reshape(((-1, 1)))

    def compute_uncertainty(self, x):
        x = torch.tensor(x)
        model_accel_func = lambda x: self.forward_acceleration(
            x,
            self.get_object_properties(x, library=torch),
            self.dists,
            self.dipole_strength,
            self.angles,
            self.rhos,
            self.zero,
        )

        model_accel = model_accel_func(x)

        model_accel_average = torch.mean(model_accel, 1)
        delta_modeled = model_accel_average.unsqueeze(1) - model_accel

        measured_acel_average = torch.mean(self.accelerations, 1)
        delta_accel = measured_acel_average.unsqueeze(1) - self.accelerations

        delta_difference = delta_accel - delta_modeled
        N = int(model_accel.shape[1])
        M = 3
        P = self.size()
        left = (delta_difference.T @ self.weights).unsqueeze(1)
        right = delta_difference.T.unsqueeze(2)
        summed_squared_delta = torch.sum(torch.bmm(left, right))
        sigma_squared = summed_squared_delta.numpy() / (N * M - P)

        big_jacobian = torch.swapaxes(
            torch.autograd.functional.jacobian(model_accel_func, x)[:, :, :, 0], 0, 1
        )
        batched_jtj = torch.bmm(torch.swapaxes(big_jacobian, 1, 2), big_jacobian)
        remap = torch.linalg.inv(torch.mean(batched_jtj, 0)).numpy()

        covariance = sigma_squared * remap / (N * M - P)
        return covariance

    def grid_sample(self, samples, scaling_function):
        possible_xs = np.linspace(self.min_bounds[0], self.max_bounds[0], samples)
        possible_ys = np.linspace(self.min_bounds[1], self.max_bounds[1], samples)
        xs, ys = np.meshgrid(possible_xs, possible_ys)
        xs = xs.flatten()
        ys = ys.flatten()
        possible_costs = [
            scaling_function(self.cost(np.stack((xs[i], ys[i])).reshape((2, 1))))
            for i in range(len(xs))
        ]
        possible_costs = np.array(possible_costs)

        best_index = np.argmin(possible_costs)
        print(
            "best brute force  func cost ",
            possible_costs[best_index],
            " at ",
            xs[best_index],
            ys[best_index],
        )

        return xs, ys, possible_costs

    def visualize_optimization(
        self,
        iterates,
        samples=50,
        func=np.abs,
        show_costs=False,
        save_location=False,
        make_heatmap=False,
    ):
        if save_location:
            font = {"weight": "bold", "size": 22}

            matplotlib.rc("font", **font)

        fig = plt.figure(figsize=(10, 10))
        if show_costs:
            ax = fig.add_subplot(projection="3d")
            start_angle = -100
            ax.view_init(elev=17.0, azim=start_angle)
        else:
            ax = fig.add_subplot()

        xs, ys, possible_costs = self.grid_sample(samples, func)

        if func is np.log:
            clip_value = np.median(possible_costs)
        else:
            clip_value = 2 * np.median(possible_costs)

        possible_costs[possible_costs > clip_value] = clip_value

        ax.set_xlim(self.min_bounds[0], self.max_bounds[0])
        ax.set_ylim(
            self.min_bounds[1] * self.sigma_multiplier,
            self.max_bounds[1] * self.sigma_multiplier,
        )
        ax.set_xlabel(self.param_names[0])
        ax.set_ylabel(self.param_names[1])

        ax.set_xticks(np.arange(0.01, 0.11, step=0.02))
        ax.set_yticks(np.arange(1e7, 8e7, step=1e7))

        points = iterates[0].reshape((-1, 2))
        costs = [
            [min(func(self.cost(point.reshape((-1, 1)))), clip_value)]
            for point in points
        ]
        if make_heatmap:
            c = ax.contourf(
                xs.reshape((samples, samples)),
                ys.reshape((samples, samples)) * self.sigma_multiplier,
                possible_costs.reshape((samples, samples)),
                30,
                cmap="RdBu",
            )
            # set the limits of the plot to the limits of the data
            ax.axis(
                [
                    self.min_bounds[0],
                    self.max_bounds[0],
                    self.min_bounds[1] * self.sigma_multiplier,
                    self.max_bounds[1] * self.sigma_multiplier,
                ]
            )
            fig.colorbar(c, ax=ax)

        if show_costs:
            ax.scatter(
                xs, ys * self.sigma_multiplier, possible_costs, color=(0, 0, 1, 0.5)
            )
            # ax.set_zlim(0, clip_value)
            points_3d = np.concatenate((points, np.array(costs)), 1)
            # silly hack to get filled circle
            scat = ax.scatter(
                points_3d[:, 0],
                points_3d[:, 1] * self.sigma_multiplier,
                points_3d[:, 2],
                linewidths=10,
                c="orange",
            )
            scat2 = ax.scatter(
                points_3d[:, 0],
                points_3d[:, 1] * self.sigma_multiplier,
                points_3d[:, 2],
                linewidths=10,
                c="orange",
            )

            # scat._offsets3d = (points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
        else:
            scat = plt.scatter(
                points[:, 0],
                points[:, 1] * self.sigma_multiplier,
                linewidths=10,
                c="orange",
            )
            scat.set_offsets(points)
            scat2 = None
        if save_location:
            fig.savefig("../../videos/initial.pdf")

        def update(num, scat, scat2, iterates, show_costs, ax, save_location):

            # ax.set_title(num)
            if show_costs:
                ax.view_init(elev=ax.elev, azim=ax.azim + 1 / 3)
            points = iterates[num].reshape((-1, 2))
            costs = [
                [min(func(self.cost(point.reshape((-1, 1)))), clip_value)]
                for point in points
            ]
            points_3d = np.concatenate((points, np.array(costs)), 1)
            # print(points_3d.shape)
            if show_costs:
                scat._offsets3d = (
                    points_3d[:, 0],
                    points_3d[:, 1] * self.sigma_multiplier,
                    points_3d[:, 2],
                )
                scat2._offsets3d = (
                    points_3d[:, 0],
                    points_3d[:, 1] * self.sigma_multiplier,
                    points_3d[:, 2],
                )

            else:
                scat.set_offsets(points * [1, self.sigma_multiplier])

            if save_location:
                fig.savefig("../../" + save_location + "/" + str(num) + ".pdf")

        anim = FuncAnimation(
            fig,
            update,
            fargs=(scat, scat2, iterates, show_costs, ax, save_location),
            frames=len(iterates),
            interval=100,
            repeat=True,
        )
        return anim

    def visualize_predicted_acceleration(self, iterate,start=0,end=-1,show_measured=True,save_location = False):
        font = {'weight' : 'bold',
                'size'   : 12}
        
        matplotlib.rc('font', **font)
        
        
        fig, ax = plt.subplots(3,1,figsize=(5, 5),sharex=True)
        font = {"weight": "bold", "size": 12}
        matplotlib.rc("font", **font)
        object_properties = self.get_object_properties(
            self.make_tensor(iterate), library=torch
        )
        modeled_accelerations = self.forward_acceleration(
            iterate,
            object_properties,
            self.dists,
            self.dipole_strength,
            self.angles,
            self.rhos,
            self.zero,
        )
        fig.suptitle("Acceleration Fit")

        titles = ["X", "Y", "θ"]
        for fig_index, acceleration_index in enumerate([0, 1, 5]):
            ax[fig_index].set_title(titles[fig_index])

            measured_acceleration = self.accelerations[acceleration_index].numpy()
            if show_measured:
                ax[fig_index].plot(measured_acceleration[start:end], label="measured")

            smoothing_factor = 10
            smoothed_acceleration = np.convolve(
                measured_acceleration,
                np.ones((smoothing_factor,)) / smoothing_factor,
                mode="same",
            )
            smooth_line = ax[fig_index].plot(
                smoothed_acceleration[start:end], label="smooth measured", linestyle="--", c=Lanc1, linewidth=3
            )

            predicted_line = ax[fig_index].plot(
                modeled_accelerations[acceleration_index][start:end], label="predicted",c=Lanc3, linewidth=3
            )
            ax[fig_index].ticklabel_format(style='sci',scilimits=(0,5))

           # if not fig_index:
          #      ax[fig_index].legend(loc="upper right")
        fig.legend([smooth_line[0],predicted_line[0]],['Measured','Predicted'])
        fig.tight_layout()
        if save_location:
            fig.savefig("../../" + save_location + "/accel.pdf")
            

    def visualize_uncertainty(self, x, multiplier=1, circles=1):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        ax.set_xlim((self.min_bounds[0], self.max_bounds[0]))
        ax.set_ylim((self.min_bounds[1], self.max_bounds[1]))
        x = x.reshape((-1, self.size()))
        for point in tqdm(x):
            point = point.reshape((self.size(), 1))
            covariance = self.compute_uncertainty(point)

            eigval, eigvec = np.linalg.eig(covariance)

            max_index = np.argmax(
                abs(eigval)
            )  # cant remember if eigvals are always positive
            angle = np.rad2deg(np.arctan2(eigvec[1, max_index], eigvec[0, max_index]))
            width = (
                np.sqrt(multiplier * eigval[max_index]) * 2
            )  # large axis   * numberSTDs * 2(radius to diamter)
            height = (
                np.sqrt(multiplier * eigval[max_index - 1]) * 2
            )  # small axis   * numberSTDs * 2(radius to diamter)
            # print(width,stdStates[i,0,0]*multiplier*2)
            for i in range(1, circles + 1):
                ells = Ellipse(
                    xy=point,
                    width=width * i,
                    height=height * i,
                    angle=angle,
                )
                ax.add_artist(ells)
                ells.set_alpha(1)
                ells.set_facecolor("none")
                ells.set_edgecolor("r")
                ells.set_clip_box(fig.axes[0].bbox)
        ax.scatter(x[:, 0], x[:, 1], linewidths=10, c="orange")


class SeperateRadiusSysid(RadiusSysid):
    def __init__(self, weights, object_properties, boat=None, drag_terms=(0, 0, 0, 0)):
        super().__init__(weights=weights, boat=boat, drag_terms=drag_terms)

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
        self.prior = self.make_tensor((self.max_bounds - self.min_bounds) / 2)

    def size(self):
        return 3

    def get_object_properties(self, x, library=np):
        radius = x[0, 0]
        sigma = x[1, 0]
        mass_radius = x[2, 0]
        mass, MOI = compute_sphere_properties(
            self.material_density, mass_radius, library
        )
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


class MassOnlySysid(SeperateRadiusSysid):
    def __init__(self, weights, object_properties, boat=None, drag_terms=(0, 0, 0, 0)):
        super().__init__(
            weights=weights, boat=boat, object_properties=object_properties
        )

        self.min_bounds[0] = 0.0
        self.min_bounds[1] = 0
        self.min_bounds[2] = 0
        self.min_bounds[3] = 0

        self.max_bounds[0] = 0.15
        self.max_bounds[1] = 0.003
        self.max_bounds[2] = 1e-2
        self.max_bounds[3] = 1e1

        self.min_bounds[4:] = 0
        self.max_bounds[4:] = 1e-3

        self.prior = self.make_tensor((self.max_bounds - self.min_bounds) / 2)
        self.initial_solution = self.prior.cpu().numpy()

        self.sigma = 5.8e7
        self.radius = 0.019
        self.boat = False

        self.regularization_multiplier = 0
        self.param_names = ["mass", "MOI_zz"]

    def get_object_properties(self, x, library=np):
        boat_mass = x[0, 0]
        boat_moi_zz = x[1:2, 0]

        if library is np:
            boat_moi_diagonal = library.concatenate((library.zeros((2)), boat_moi_zz))
        else:
            boat_moi_diagonal = library.concat((library.zeros((2)), boat_moi_zz))

        boat_moi = library.diag(boat_moi_diagonal)

        mass, MOI = compute_sphere_properties(
            self.material_density, self.radius, library=library
        )
        object_properties = ObjectProperties(
            mass=mass + boat_mass,
            MOI=MOI + boat_moi,
            radius=self.radius,
            sigma=self.sigma,
        )

        return object_properties

    def size(self):
        return 6

    def get_drag_wrenches(self, x):
        linear_drag_force_constant = x[2, 0]
        quadratic_drag_force_constant = x[3, 0]

        linear_drag_torque_constant = x[4, 0]
        quadratic_drag_torque_constant = x[5, 0]

        drag_force = (
            linear_drag_force_constant * self.linear_velocity
            + torch.sign(self.linear_velocity)
            * quadratic_drag_force_constant
            * self.linear_velocity**2
        )

        drag_torque = (
            linear_drag_torque_constant * self.rotational_velocity
            + torch.sign(self.rotational_velocity)
            * quadratic_drag_torque_constant
            * self.rotational_velocity**2
        )

        drag_wrench = torch.concat((-drag_force, -drag_torque), 1)
        return drag_wrench


class MassSysID(SeperateRadiusSysid):
    def __init__(
        self, weights, object_properties=None, boat=None, drag_terms=(0, 0, 0, 0)
    ):
        super().__init__(
            weights=weights,
            object_properties=object_properties,
            boat=boat,
            drag_terms=drag_terms,
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

        # self.min_bounds[0] = 1e-3

        # self.min_bounds[1] = 1e4 / self.sigma_multiplier

        # radius and sigma keep the same
        # self.min_bounds[2] = 1e-10
        # self.max_bounds[2] = 2

        self.min_bounds[2] = -1  # mass
        self.min_bounds[3:] = -1e-3  # MOI

        self.max_bounds[2] = 2  # mass
        self.max_bounds[3:] = 1e-3  # MOI
        self.prior = self.make_tensor((self.max_bounds - self.min_bounds) / 2)

    def size(self):
        return 6

    def get_object_properties(self, x, library=np):
        radius = x[0, 0]
        sigma = x[1, 0]
        mass_extra = x[2, 0]
        moi_extra = library.diag(x[3:, 0])

        mass_sphere, moi_sphere = compute_sphere_properties(
            self.material_density, radius, library
        )

        mass = mass_sphere + mass_extra
        MOI = moi_sphere + moi_extra

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
    prob = sys_id_type(
        build_acceleration_q("all"), boat
    )  # built with M(0.04) Q M(0.04)
    optimizer = NewtonMethod(problem=prob, **optimizer_params)
    prob.updateVals(controls, magnets, positions, linear_velocity, rotational_velocity)
    result = optimizer.optimize(prob.initial_solution, max_iterations=50)
    prob.visualize_optimization(result.iterates)
    return result
