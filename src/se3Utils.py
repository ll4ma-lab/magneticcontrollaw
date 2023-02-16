# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:48:58 2021

@author: Griffin Tabor
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def build_skew(vector):
    vector = np.reshape(vector, (3))
    skew = np.array(
        [
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0],
        ]
    )
    return skew


def matrix_log(Ae):
    if np.abs(np.trace(Ae) - 1) > 2.0:
        Ae = np.copy(Ae)
        Ae[:, 0] /= np.linalg.norm(Ae[:, 0])  # unit vectorize because numerical issues
        Ae[:, 1] /= np.linalg.norm(Ae[:, 1])
        Ae[:, 2] /= np.linalg.norm(Ae[:, 2])
    phi = np.arccos(0.5 * (np.trace(Ae) - 1))
    skew = (1 / (2 * np.sin(phi))) * (Ae - np.transpose(Ae))
    return skew, phi


def matrix_exponential(w_hat_skew, theta):
    rot = (
        np.eye(3)
        + np.sin(theta) * w_hat_skew
        + (1 - np.cos(theta)) * np.dot(w_hat_skew, w_hat_skew)
    )
    return rot


def repair_rot_matrix(transform):
    skew, phi = matrix_log(transform[:3, :3])
    vec = skew_to_vector(skew)
    skew /= np.linalg.norm(vec)
    rot = matrix_exponential(skew, phi)
    new_transform = np.copy(transform)
    new_transform[:3, :3] = rot[:3, :3]
    if np.any(np.isnan(new_transform)):
        # usually means no reasonable twist
        return transform
    return new_transform


def skew_to_vector(skew):
    w = np.zeros((3, 1))
    w[0, 0] = skew[2, 1]
    w[1, 0] = skew[0, 2]
    w[2, 0] = skew[1, 0]
    return w


def build_tf(position, angles):
    world_to_body_rotation = R.from_euler("xyz", angles)
    world_to_body_r = world_to_body_rotation.as_matrix()

    transform = np.eye(4)
    transform[:3, :3] = world_to_body_r
    transform[:3, 3] = position

    return transform
