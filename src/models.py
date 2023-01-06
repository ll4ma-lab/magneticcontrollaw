#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 17:23:35 2021

@author: tabor
"""
import numpy as np
import torch
import time
from dataclasses import dataclass


@dataclass
class CylinderControlCommand:
    mag_num: int = 0
    model_num: int = 0
    dipole_strength: float = 0.0
    theta: float = 0
    seconds: float = 1.0


@dataclass
class SphereControlCommand:
    mag_num: int = 0
    dipole_strength: float = 0.0
    psi: float = 0
    xi: float = 0
    seconds: float = 1.0


# cylinder model helper
def get_omni_frame(pos):
    z = np.array([0, 0, 1])
    dist = np.linalg.norm(pos)

    omni_z = -1 * (pos / dist)
    omni_y = np.cross(omni_z, z)
    omni_y = omni_y / np.sqrt(np.dot(omni_y, omni_y))  # unit vectorise
    omni_x = np.cross(omni_y, omni_z)
    # Combine to build rotation matrix
    omni_frame = np.transpose(np.array([omni_x, omni_y, omni_z]))
    return omni_frame


def average_dict(a, b):
    new_dict = {}
    for key in a.keys():
        new_dict[key] = ((np.array(a[key]) + np.array(b[key])) / 2).tolist()
    return new_dict


# cylinder model helper
def rot_z(yaw, library=np):
    yaw = yaw * np.pi / 180.0
    one, zero = (
        (1.0, 0.0) if library is np else (torch.tensor([1.0]), torch.tensor([0.0]))
    )

    r_z = library.stack(
        [
            library.stack([library.cos(yaw), -library.sin(yaw), zero]),
            library.stack([library.sin(yaw), library.cos(yaw), zero]),
            library.stack([zero, zero, one]),
        ]
    ).reshape(3, 3)
    return r_z


# sphere model helper
def construct_omega(angles, library=np):
    angles = angles * np.pi / 180.0
    omega = library.stack(
        [
            library.sin(angles[0]) * library.cos(angles[1]),
            library.sin(angles[0]) * library.sin(angles[1]),
            library.cos(angles[0]),
        ]
    ).reshape(3, -1)
    return omega


def deconstruct_omega(omega):
    omega = omega * (1 - 1e-15)
    first_angle = np.arccos(omega[2])
    if 1 - omega[2] ** 2 < 1e-3:
        second_angle = 0
    else:
        sin_angle2 = omega[1] / np.sin(first_angle)
        cos_angle2 = omega[0] / np.sin(first_angle)
        second_angle = np.arctan2(sin_angle2, cos_angle2)
    return np.array([first_angle, second_angle]) / (np.pi / 180.0)


def vector_norm(vector, library):
    if library is np:
        norm = library.linalg.norm(vector, axis=1)  # library agnostic L2 norm
    else:
        norm = library.linalg.norm(vector, dim=1)  # library agnostic L2 norm
    return norm


# sphere model helper
def compute_theta_and_frame(omega, rho, library=np):

    omega = library.swapaxes(omega, 0, 1)
    rho = library.swapaxes(rho, 0, 1)  # libraries dont like 3,n and prefer n,3

    cross = library.cross(omega, rho)

    l2_cross = vector_norm(cross, library)

    dot = library.sum(omega * rho, 1)
    if library is np:
        theta = library.arctan2(l2_cross, dot)
    else:
        theta = library.atan2(l2_cross, dot)
    omni_z = (rho.T / vector_norm(rho, library)).T
    omni_y = (cross.T / l2_cross).T

    omni_x = library.cross(omni_y, omni_z)
    # Combine to build rotation matri
    stacked = library.stack([omni_x, omni_y, omni_z])
    # first move the batch dimension to the end, then tranpose the rotation matrix
    omni_frame = library.swapaxes(stacked, 0, 1)
    omni_frame = library.swapaxes(omni_frame, 1, 2)

    return theta, omni_frame


# sphere model helper
def model_call_rotated_wrench(
    dipole_strength, angles, model_vals, rho, library=np, zero=np.array([0])
):
    frho0, trho0, frho90, fphi90, ttheta90 = model_vals
    omega = construct_omega(angles, library)
    theta, rotation = compute_theta_and_frame(omega, rho, library)
    unit_strength_wrench = library.stack(
        [
            zero,
            fphi90 * library.sin(theta),
            (-(frho90 - frho0) / 2) * library.cos(2 * theta) + (frho90 + frho0) / 2,
            ttheta90 * library.sin(theta),
            zero,
            trho0 * library.cos(theta),
        ]
    )
    # switch batch dim to 0
    unit_strength_wrench = library.swapaxes(unit_strength_wrench, 0, 1)
    if library is np:
        unit_strength_wrench = np.expand_dims(unit_strength_wrench, 2)
        selected_wrench = [
            dipole_strength[i] ** 2
            * np.concatenate(
                (
                    rotation[i] @ unit_strength_wrench[i, :3],
                    rotation[i] @ unit_strength_wrench[i, 3:],
                )
            )
            for i in range(unit_strength_wrench.shape[0])
        ]
        selected_wrench = np.array(selected_wrench)
    else:
        unit_strength_wrench = torch.unsqueeze(unit_strength_wrench, 2)

        if dipole_strength.shape[0] == 1:
            selected_wrench = torch.cat(
                (
                    rotation[0] @ unit_strength_wrench[0, :3],
                    rotation[0] @ unit_strength_wrench[0, 3:],
                ),
                0,
            ).T
            selected_wrench = torch.unsqueeze(selected_wrench * dipole_strength**2, 2)
        else:
            forces = torch.bmm(rotation, unit_strength_wrench[:, :3])
            torques = torch.bmm(rotation, unit_strength_wrench[:, 3:])
            selected_wrench = torch.bmm(
                torch.cat((forces, torques), 1), dipole_strength.reshape((-1, 1, 1)) ** 2
            )

    return selected_wrench


class MultiMagModel:
    def __init__(self, magnets, object_properties, use_sphere=False):
        self.time = 0
        self.magnets = magnets
        self.object_properties = object_properties
        self.cylinder_model = CylinderModel(object_properties)
        self.sphere_model = SphereModel(object_properties)
        self.use_sphere = use_sphere

    def forward_pass(self, position, control):
        if self.use_sphere:
            return self.forward_sphere(position, control)
        else:
            return self.forward_cylinder(position, control)

    def forward_cylinder(self, position, control):
        s = time.time()
        chosen_magnet_pose = (np.array(self.magnets) - np.transpose(position)).tolist()[
            control.mag_num
        ]
        omni_frame = get_omni_frame(chosen_magnet_pose)

        if control.model_num == 2:
            rotation = np.dot(omni_frame, rot_z(control.theta))
        else:
            rotation = omni_frame

        double_rot_matrix = np.zeros((6, 6))
        double_rot_matrix[:3, :3] = rotation
        double_rot_matrix[3:, 3:] = rotation

        wrench = self.cylinder_model.model_call(
            np.linalg.norm(chosen_magnet_pose), control.dipole_strength
        )[control.model_num]

        rotated_wrench = double_rot_matrix @ wrench
        self.time += time.time() - s
        return rotated_wrench

    def forward_sphere(self, position, control):
        s = time.time()
        chosen_magnet_pose = (
            np.array(self.magnets)[control.mag_num] - np.transpose(position)
        ).transpose()
        dist = np.linalg.norm(chosen_magnet_pose)
        rho = (-1 * chosen_magnet_pose).reshape((3, 1))

        model_vals = self.sphere_model.model_call(dist, 1)
        wrench = model_call_rotated_wrench(
            np.array(control.dipole_strength).reshape(1),
            np.array((control.psi, control.xi)),
            model_vals,
            rho,
        ).reshape((6, 1))

        self.time += time.time() - s
        return wrench


class SphereModel:
    def __init__(self, object_properties):
        self.cylinder_model = CylinderModel(object_properties)

    def model_call(self, dist, dipole_strength, radii=None, sigma=None):
        frho0 = self.cylinder_model.model_equation(
            self.cylinder_model.constants["fzz"], dist, dipole_strength, 4, radii, sigma
        )
        trho0 = self.cylinder_model.model_equation(
            self.cylinder_model.constants["tzz"], dist, dipole_strength, 3, radii, sigma
        )

        frho90 = self.cylinder_model.model_equation(
            self.cylinder_model.constants["frhorho"],
            dist,
            dipole_strength,
            4,
            radii,
            sigma,
        )
        fphi90 = self.cylinder_model.model_equation(
            self.cylinder_model.constants["frhophi"],
            dist,
            dipole_strength,
            4,
            radii,
            sigma,
        )
        ttheta90 = self.cylinder_model.model_equation(
            self.cylinder_model.constants["trhoz"],
            dist,
            dipole_strength,
            3,
            radii,
            sigma,
        )
        return frho0, trho0, frho90, fphi90, ttheta90


class CylinderModel:
    def __init__(self, object_properties):
        self.time = 0.0

        self.object_properties = object_properties
        self.freq = 15
        self.mu = 4 * np.pi * 1e-7

        self.fea_constants = {}
        self.fea_constants["fzz"] = [430, 2.95, -0.101, -9.26, 7]
        self.fea_constants["tzz"] = [6840, 3.00, -0.0986, -13.2, 6]
        self.fea_constants["frhorho"] = [266, 2.60, -0.101, -7.65, 7]
        self.fea_constants["frhophi"] = [6040, 3.45, -0.102, -14.3, 7]
        self.fea_constants["trhoz"] = [8100, 3.60, -0.0985, -15.7, 6]

        self.exp_constants = {}
        self.exp_constants["fzz"] = [467, 2.81, -0.0969, -9.75, 7]
        self.exp_constants["tzz"] = [6900, 3.35, -0.0990, -14.9, 6]
        self.exp_constants["frhorho"] = [282, 3.20, -0.0980, -9.41, 7]
        self.exp_constants["frhophi"] = [5870, 3.49, -0.0973, -14.6, 7]
        self.exp_constants["trhoz"] = [8000, 3.40, -0.0928, -15.0, 6]

        self.constants = average_dict(self.fea_constants, self.exp_constants)

    def model_equation(
        self, constants, distance, dipole_strength, p0_r_power, radii=None, sigma=None
    ):
        if radii is None:
            radii = self.object_properties.radius
        if sigma is None:
            sigma = self.object_properties.sigma
        # P0 has r^4 for forces and r^3 for torques
        p1 = sigma * self.mu * self.freq * radii * radii
        p2 = distance / radii

        numerator_base = constants[0] * p1
        numerator_power = constants[1] * (p1 ** constants[2])
        numerator = (numerator_base**numerator_power) * (10 ** constants[3])

        denominator = p2 ** constants[4]
        p0 = numerator / denominator

        c = self.mu * (dipole_strength**2) * (radii**-p0_r_power)

        wrench = p0 * c

        return wrench

    def model_call(self, dist, dipole_strength):
        fzz = self.model_equation(self.constants["fzz"], dist, dipole_strength, 4)
        tzz = self.model_equation(self.constants["tzz"], dist, dipole_strength, 3)

        frhorho = self.model_equation(self.constants["frhorho"], dist, dipole_strength, 4)
        frhophi = self.model_equation(self.constants["frhophi"], dist, dipole_strength, 4)
        trhoz = self.model_equation(self.constants["trhoz"], dist, dipole_strength, 3)

        pointing_model = np.array([0, 0, fzz, 0, 0, tzz]).reshape((6, 1))
        neg_pointing_model = np.array([0, 0, fzz, 0, 0, -tzz]).reshape((6, 1))

        rho_model = np.array([0, -frhophi, frhorho, -trhoz, 0, 0]).reshape((6, 1))
        return pointing_model, neg_pointing_model, rho_model
