# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:24:22 2019

@author: tabor
"""

import numpy as np
from se3Utils import build_skew, build_tf, matrix_log, skew_to_vector


class PDController:
    def __init__(
        self,
        state,
        linear_kp,
        rotational_kp,
        linear_kd,
        rotational_kd,
        linear_ffd=0.0,
        rotational_ffd=0.0,
    ):
        self.last_error = None
        dims = state.shape[0]
        half_dims = int(dims / 2)
        quarter_dims = int(half_dims / 2)
        self.pgains = np.zeros((half_dims, half_dims))
        self.dgains = np.zeros((half_dims, half_dims))
        self.ff_dgains = np.zeros((half_dims, half_dims))

        self.pgains[:quarter_dims, :quarter_dims] = np.eye(quarter_dims) * linear_kp
        self.pgains[quarter_dims:, quarter_dims:half_dims] = (
            np.eye(quarter_dims) * rotational_kp
        )

        self.dgains[:quarter_dims, :quarter_dims] = np.eye(quarter_dims) * linear_kd
        self.dgains[quarter_dims:, quarter_dims:half_dims] = (
            np.eye(quarter_dims) * rotational_kd
        )

        self.ff_dgains[:quarter_dims, :quarter_dims] = np.eye(quarter_dims) * linear_ffd
        self.ff_dgains[quarter_dims:, quarter_dims:half_dims] = (
            np.eye(quarter_dims) * rotational_ffd
        )

        self.last_desired = None

    def compute_error(self, current_transform, desired_transform):
        error_transform = np.dot(np.linalg.inv(current_transform), desired_transform)
        linear_error = error_transform[:3, 3:]
        skew, theta = matrix_log(error_transform[:3, :3])
        if theta == 0.0:
            rotation_error = np.zeros((3, 1))
        else:
            w_hat = skew_to_vector(skew)
            rotation_error = w_hat * theta

        linear_error_w = np.dot(current_transform[:3, :3], linear_error)
        rotation_error_w = np.dot(current_transform[:3, :3], rotation_error)
        return np.concatenate((linear_error_w, rotation_error_w))

    def get_feedforward_component(self,current,desired):
        f_fstuff = np.zeros((6,1))
        if self.last_desired is not None:
            ff_error = desired - self.last_desired
            f_fstuff = np.matmul(self.ff_dgains, ff_error[6:])
        self.last_desired = np.copy(desired)
        return f_fstuff
            
    def get_desired_wrench(self, current, desired):

        current_tf = current[0]
        desired_tf = build_tf(desired[:3, 0], desired[3:6, 0])

        position_error = self.compute_error(current_tf, desired_tf)
        # (bodyToWorld,Ωsc,Vsc)

        rot_vel_error = desired[9:12] - current_tf[:3, :3] @ current[1]
        linear_vel_error = desired[6:9] - current_tf[:3, :3] @ current[2]
        vel_error = np.concatenate((linear_vel_error, rot_vel_error), 0)

        pstuff = np.matmul(self.pgains, position_error)
        dstuff = np.matmul(self.dgains, vel_error)

        total = pstuff + dstuff
        f_fstuff = self.get_feedforward_component(current,desired)
        total += f_fstuff

        return total, pstuff, dstuff
