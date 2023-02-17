# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:20:28 2020

@author: tabor
"""
import numpy as np
from enum import Enum


class PlanarEnvironments(Enum):
    LINE = 1
    SQUARE = 2
    DIAMOND = 3
    DOUBLE_SQUARE = 4
    SQUARE_FIXED_HEADING = 5
    ROTATE_IN_PLACE = 6
    ROTATE_IN_PLACE_FAST = 12
    SQUARE_ROTATE = 7
    SQUARE_ROTATE_EASY = 11
    INFINITY = 8
    OMEGA = 9
    U = 10


class SpaceEnvironments(Enum):
    SQUARE = 1
    AXIS = 2
    SPIRAL = 3
    REGULATOR = 4
    CUBE = 5
    CUBE_ROTATE = 6
    SPIN = 7
    SPIN2 = 8
    BACKANDFORTH = 9


class PlanarProblem:
    def __init__(self, challenge_id=0):
        self.trajectory_dimensions = "xyt"
        width = 0.38
        length = 0.62
        height = 0.15
        self.visualization_boundary = (-length / 2, length / 2)
        self.visualization_boundary_x = (-length / 2, length / 2)
        self.visualization_boundary_y = (-width / 2, width / 2)
        self.visualization_boundary_z = (-width / 2, width / 2)
        self.az = -90
        self.el = -90
        self.lims = [
            (-width / 2, width / 2),
            (-length / 2, length / 2),
            (-height / 2, height / 2),
        ]

        z_value = 0.15
        magnets = np.array(
            [
                (0.1, 0.1, z_value),  # marker 0.365m away
                (-0.1, 0.1, z_value),
                (-0.1, -0.1, z_value),
                (0.1, -0.1, z_value),
            ]
        )
        self.magnets = magnets.tolist()

        self.radius = 0.065
        linear_tolerance = 0.005
        deg_tolerance = 5.0
        rotational_tolerance = np.pi * deg_tolerance / 180.0

        self.tolerances = (linear_tolerance, rotational_tolerance)

        self.start_state = np.zeros((12, 1))
        square_radius = 0.075
        if challenge_id == PlanarEnvironments.LINE:  # line
            self.true_desired_state = np.zeros((12, 2))
            self.true_desired_state[:3, 0] = [square_radius, -square_radius, 0]
            # self.TrueDesiredState[:3,1] = [0.075,-0.075,0]
            self.true_desired_state[:3, 1] = [-square_radius, -square_radius, 0]
            self.desired_state = self.true_desired_state
            self.times = [180] * self.desired_state.shape[1]
            self.times[0] = 180
        if (
            challenge_id == PlanarEnvironments.SQUARE
            or challenge_id == PlanarEnvironments.DIAMOND
        ):  # square
            self.true_desired_state = np.zeros((12, 5))
            self.true_desired_state[:3, 0] = [square_radius, -square_radius, 0]
            self.true_desired_state[:3, 1] = [-square_radius, -square_radius, 0]
            self.true_desired_state[:3, 2] = [-square_radius, square_radius, 0]
            self.true_desired_state[:3, 3] = [square_radius, square_radius, 0]
            self.true_desired_state[:3, 4] = [square_radius, -square_radius, 0]

            self.desired_state = self.true_desired_state
            self.times = [240] * self.desired_state.shape[1]
            self.trajectory_dimensions = "xy"

        if challenge_id == PlanarEnvironments.DOUBLE_SQUARE:  # square
            self.true_desired_state = np.zeros((12, 9))
            self.true_desired_state[:3, 0] = [square_radius, -square_radius, 0]
            self.true_desired_state[:3, 1] = [-square_radius, -square_radius, 0]
            self.true_desired_state[:3, 2] = [-square_radius, square_radius, 0]
            self.true_desired_state[:3, 3] = [square_radius, square_radius, 0]
            self.true_desired_state[:3, 4] = [square_radius, -square_radius, 0]

            self.true_desired_state[:3, 5] = [-square_radius, -square_radius, 0]
            self.true_desired_state[:3, 6] = [-square_radius, square_radius, 0]
            self.true_desired_state[:3, 7] = [square_radius, square_radius, 0]
            self.true_desired_state[:3, 8] = [square_radius, -square_radius, 0]

            self.desired_state = self.true_desired_state
            self.times = [240] * self.desired_state.shape[1]
            self.trajectory_dimensions = "xy"

        if (
            challenge_id == PlanarEnvironments.SQUARE_FIXED_HEADING
        ):  # square fixed rotation
            self.true_desired_state = np.zeros((12, 5))
            self.true_desired_state[:3, 0] = [square_radius, -square_radius, 0]
            self.true_desired_state[:3, 1] = [-square_radius, -square_radius, 0]
            self.true_desired_state[:3, 2] = [-square_radius, square_radius, 0]
            self.true_desired_state[:3, 3] = [square_radius, square_radius, 0]
            self.true_desired_state[:3, 4] = [square_radius, -square_radius, 0]

            self.desired_state = self.true_desired_state
            self.times = [360] * self.desired_state.shape[1]
        if challenge_id == PlanarEnvironments.DIAMOND:  # diamond
            angle = 45 / (180 / np.pi)
            R = 0.75 * np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )

            for i in range(5):
                self.true_desired_state[:2, i] = np.dot(
                    R, self.true_desired_state[:2, i]
                )

            self.desired_state = self.true_desired_state
            self.times = [360] * self.desired_state.shape[1]
            self.trajectory_dimensions = "xyt"

        if challenge_id == PlanarEnvironments.ROTATE_IN_PLACE:  # stationary rotation
            self.true_desired_state = np.zeros((12, 5))
            # self.TrueDesiredState[1,:] = -0.075

            self.true_desired_state[5, 0] = -np.pi / 2
            self.true_desired_state[5, 1] = 0
            self.true_desired_state[5, 2] = np.pi / 2
            self.true_desired_state[5, 3] = np.pi
            self.true_desired_state[5, 4] = 0

            # self.TrueDesiredState[4,:] = np.pi/2

            self.desired_state = self.true_desired_state
            self.times = [360] * self.desired_state.shape[1]
            self.times[0] = 480

        if (
            challenge_id == PlanarEnvironments.ROTATE_IN_PLACE_FAST
        ):  # stationary rotation
            seperate_steps = 100
            self.true_desired_state = np.zeros((12, seperate_steps))

            a = 0.0001
            time_per_target = 30
            for t in range(1, seperate_steps + 1):
                self.true_desired_state[5, t - 1] = (a / 2) * (
                    (t * time_per_target) ** 2
                )
                self.true_desired_state[11, t - 1] = a * (t * time_per_target)
            # self.TrueDesiredState[4,:] = np.pi/2

            self.desired_state = self.true_desired_state
            self.times = [time_per_target] * self.desired_state.shape[1]

        if challenge_id == PlanarEnvironments.SQUARE_ROTATE:  # rotate while square
            self.start_state[5] = -np.pi / 4

            self.true_desired_state = np.zeros((12, 10))
            self.true_desired_state[5, 0] = -np.pi / 4

            self.true_desired_state[:3, 0] = [square_radius, -square_radius, 0]
            self.true_desired_state[:3, 1] = self.true_desired_state[:3, 0]
            self.true_desired_state[5, 1] = -np.pi
            self.true_desired_state[5, 2] = -np.pi

            self.true_desired_state[:3, 2] = [-square_radius, -square_radius, 0]
            self.true_desired_state[:3, 3] = self.true_desired_state[:3, 2]
            self.true_desired_state[5, 3] = -3.0 * np.pi / 2.0
            self.true_desired_state[5, 4] = -3.0 * np.pi / 2.0

            self.true_desired_state[:3, 4] = [-square_radius, square_radius, 0]
            self.true_desired_state[:3, 5] = self.true_desired_state[:3, 4]
            self.true_desired_state[5, 5] = -2.0 * np.pi
            self.true_desired_state[5, 6] = -2.0 * np.pi

            self.true_desired_state[:3, 6] = [square_radius, square_radius, 0]
            self.true_desired_state[:3, 7] = self.true_desired_state[:3, 6]
            self.true_desired_state[5, 7] = -2.5 * np.pi
            self.true_desired_state[5, 8] = -2.5 * np.pi

            self.true_desired_state[:3, 8] = [square_radius, -square_radius, 0]
            self.true_desired_state[:3, 9] = self.true_desired_state[:3, 8]
            self.true_desired_state[5, 9] = -3.0 * np.pi

            self.desired_state = self.true_desired_state
            self.times = [240] * self.desired_state.shape[1]

        if challenge_id == PlanarEnvironments.SQUARE_ROTATE_EASY:  # rotate while square
            self.start_state[5] = -np.pi / 4
            self.start_state[:3, 0] = [square_radius, -square_radius, 0]

            self.true_desired_state = np.zeros((12, 9))
            self.true_desired_state[5, 0] = -np.pi / 4

            self.true_desired_state[:3, 0] = [square_radius, -square_radius, 0]
            self.true_desired_state[5, 0] = -np.pi
            self.true_desired_state[5, 1] = -np.pi

            self.true_desired_state[:3, 1] = [-square_radius, -square_radius, 0]
            self.true_desired_state[:3, 2] = self.true_desired_state[:3, 1]
            self.true_desired_state[5, 2] = -3.0 * np.pi / 2.0
            self.true_desired_state[5, 3] = -3.0 * np.pi / 2.0

            self.true_desired_state[:3, 3] = [-square_radius, square_radius, 0]
            self.true_desired_state[:3, 4] = self.true_desired_state[:3, 3]
            self.true_desired_state[5, 4] = -2.0 * np.pi
            self.true_desired_state[5, 5] = -2.0 * np.pi

            self.true_desired_state[:3, 5] = [square_radius, square_radius, 0]
            self.true_desired_state[:3, 6] = self.true_desired_state[:3, 5]
            self.true_desired_state[5, 6] = -2.5 * np.pi
            self.true_desired_state[5, 7] = -2.5 * np.pi

            self.true_desired_state[:3, 7] = [square_radius, -square_radius, 0]
            self.true_desired_state[:3, 8] = self.true_desired_state[:3, 7]
            self.true_desired_state[5, 8] = -3.0 * np.pi

            self.desired_state = self.true_desired_state
            self.times = [480] * self.desired_state.shape[1]

        if challenge_id == PlanarEnvironments.INFINITY:  # infinity
            vel = 0.0005

            self.true_desired_state = np.zeros((12, 8))
            self.true_desired_state[:3, 0] = [-0.06, 0.04, 0]
            self.true_desired_state[6:9, 0] = [-vel, 0, 0]

            self.true_desired_state[:3, 1] = [-0.12, 0, 0]  # left
            self.true_desired_state[6:9, 1] = [0, -vel / 2, 0]

            self.true_desired_state[:3, 2] = [-0.06, -0.04, 0]
            self.true_desired_state[6:9, 2] = [vel, 0, 0]

            self.true_desired_state[:3, 3] = [-0.0, 0.0, 0]  # center

            self.true_desired_state[:3, 4] = [0.06, 0.04, 0]
            self.true_desired_state[6:9, 4] = [vel, 0, 0]

            self.true_desired_state[:3, 5] = [0.12, 0, 0]  # left
            self.true_desired_state[6:9, 5] = [0, -vel / 2, 0]

            self.true_desired_state[:3, 6] = [0.06, -0.04, 0]
            self.true_desired_state[6:9, 6] = [-vel, 0, 0]

            self.true_desired_state[:3, 7] = [-0.0, 0.0, 0]  # center

            self.desired_state = self.true_desired_state
            self.times = [180] * self.desired_state.shape[1]
            self.times[0] = 180
            # self.trajectoryDimensions = 'xy'
        if challenge_id == PlanarEnvironments.OMEGA:  # U
            vel = 0.1
            self.true_desired_state = np.zeros((12, 15))
            self.true_desired_state[:3, 0] = [5, -7, 0]
            self.true_desired_state[:3, 1] = [3, -7, 0]
            self.true_desired_state[:3, 2] = [3, -11, 0]
            self.true_desired_state[:3, 3] = [13, -11, 0]
            self.true_desired_state[:3, 4] = [13, -7, 0]
            self.true_desired_state[:3, 5] = [11, -7, 0]

            self.true_desired_state[:3, 6] = [0, 13, 0]
            self.true_desired_state[6:9, 6] = [-0.2, 0, 0]

            self.true_desired_state[:3, 7] = [-11, -7, 0]
            self.true_desired_state[:3, 8] = [-13, -7, 0]
            self.true_desired_state[:3, 9] = [-13, -11, 0]
            self.true_desired_state[:3, 10] = [-3, -11, 0]
            self.true_desired_state[:3, 11] = [-3, -7, 0]
            self.true_desired_state[:3, 12] = [-5, -7, 0]

            self.true_desired_state[:3, 13] = [-0, 9, 0]
            self.true_desired_state[:3, 14] = [5, -7, 0]
            self.true_desired_state[6:9, 13] = [0.08, 0, 0]

            self.true_desired_state = self.true_desired_state * 0.005

            self.desired_state = self.true_desired_state
            self.times = [240] * self.desired_state.shape[1]
            self.times[6] = 360
            self.times[7] = 360

            self.times[13] = 360
            self.times[14] = 360

        if challenge_id == PlanarEnvironments.U:  # block U
            self.true_desired_state = np.zeros((12, 18))
            self.true_desired_state[:2] = (
                np.array(
                    [
                        [721, 30],
                        [30, 30],
                        [30, 446],
                        [169, 447],
                        [170, 1129],
                        [454, 1411],
                        [1133, 1411],
                        [1417, 1129],
                        [1418, 447],
                        [1550, 446],
                        [1550, 30],
                        [859, 30],
                        [859, 446],
                        [998, 447],
                        [998, 993],
                        [582, 993],
                        [582, 447],
                        [721, 446],
                    ]
                ).T
                - np.array([[750, 750]]).T
            )

            short_time = 240
            long_time = 240

            self.desired_state = self.true_desired_state * 0.0001
            self.start_state[:2, 0] = self.desired_state[:2, -1]

            self.times = [long_time] * self.desired_state.shape[1]
            self.times[3] = short_time
            self.times[17] = short_time
            self.times[13] = short_time
            self.times[9] = short_time

            # self.trajectoryDimensions = 'xy'
        import warnings

        if len(self.trajectory_dimensions) < 3:
            warnings.warn(
                "planar environment without yaw control can \
                have numerical issues as yaw vel gets high"
            )

    def problem_specific_modification(self, state, wrench=None):

        # note state has TF and body frame velocity, trying to modify vel
        # means probably rotating to world frame
        body_to_world = state[0][:3, :3]
        linear_velocity = np.dot(body_to_world, state[2])
        rotational_velocity = np.dot(body_to_world, state[1])

        world_to_body = np.linalg.inv(body_to_world)

        linear_velocity[2] = 0
        rotational_velocity[0] = 0
        rotational_velocity[1] = 0

        state[2][:] = np.dot(world_to_body, linear_velocity)
        state[1][:] = np.dot(world_to_body, rotational_velocity)

        if wrench is not None:
            wrench[2] = 0  # z force
            wrench[3] = 0  # roll torque
            wrench[4] = 0  # pitch torque


class SpaceProblem:
    def __init__(
        self,
        challenge_id=0,
    ):
        self.trajectory_dimensions = "xyzrpt"
        width = 0.001
        length = 0.001
        height = 0.001
        self.visualization_boundary_x = (-0.3, 0.3)
        self.visualization_boundary_y = (-0.3, 0.3)
        self.visualization_boundary_z = (-0.3, 0.3)
        self.az = -43
        self.el = -160
        self.lims = [
            (-width / 2, width / 2),
            (-length / 2, length / 2),
            (-height / 2, height / 2),
        ]

        linear_tolerance = 0.005
        deg_tolerance = 10.0
        rotational_tolerance = np.pi * deg_tolerance / 180.0

        self.tolerances = (linear_tolerance, rotational_tolerance)
        self.radius = 0.065

        num_pts = 6
        from numpy import pi, cos, sin, arccos, arange

        indices = arange(0, num_pts, dtype=float) + 0.5

        phi = arccos(1 - 2 * indices / num_pts)
        theta = pi * (1 + 5**0.5) * indices

        x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)
        magnets = np.transpose(np.array((x, y, z))) * 0.15

        magnets = np.array(
            [
                (0.15, 0.00, 0),
                (-0.15, 0.0, 0),
                (0.00, 0.150, 0.0),
                (0.00, -0.150, 0.0),
                (1e-10, 0.000, 0.15),
                (1e-10, 0.00, -0.15),
            ]
        )
        self.magnets = magnets.tolist()
        self.start_state = np.zeros((12, 1))

        if challenge_id == SpaceEnvironments.SQUARE:  # floating square
            self.true_desired_state = np.zeros((12, 5))
            self.true_desired_state[:3, 0] = [0.075, -0.075, -0.075]
            self.true_desired_state[:3, 1] = [-0.075, -0.075, -0.075]
            self.true_desired_state[:3, 2] = [-0.075, 0.075, 0.075]
            self.true_desired_state[:3, 3] = [0.075, 0.075, 0.075]
            self.true_desired_state[:3, 4] = [0.075, -0.075, 0]

            self.desired_state = self.true_desired_state
            self.times = [480] * self.desired_state.shape[1]

        if challenge_id == SpaceEnvironments.AXIS:  # individual axis
            self.true_desired_state = np.zeros((12, 12))
            self.true_desired_state[0, 0] = 0.1

            self.true_desired_state[1, 2] = 0.1

            self.true_desired_state[2, 4] = 0.1

            self.true_desired_state[3, 6] = np.pi / 2

            self.true_desired_state[4, 8] = np.pi / 2

            self.true_desired_state[5, 10] = np.pi / 2

            self.desired_state = self.true_desired_state
            self.times = [180] * self.desired_state.shape[1]
        if challenge_id == SpaceEnvironments.SPIRAL:  # spiral WIP
            i = 0
            self.start_state[:, 0] = self.spiral_helper(0)
            components = 40
            self.true_desired_state = np.zeros((12, components))
            for i in range(components):
                self.true_desired_state[:, i] = self.spiral_helper(
                    i * 2 * np.pi / components
                )
            self.desired_state = self.true_desired_state
            self.times = [120] * self.desired_state.shape[1]
        if challenge_id == SpaceEnvironments.REGULATOR:
            self.start_state[:3, 0] = np.linspace(0.05, 0.05, 3)
            self.start_state[3:6, 0] = np.linspace(0.5, 1.5, 3)

            self.true_desired_state = np.zeros((12, 1))

            self.desired_state = self.true_desired_state
            self.times = [600]
        if challenge_id == SpaceEnvironments.CUBE:  # floating square
            self.magnets = [self.magnets[i] for i in [0, 1]]
            self.magnets[0][1] += 1e-15
            self.magnets[1][1] += 1e-15
            self.true_desired_state = np.zeros((12, 9))
            r = 0.025
            self.true_desired_state[:3, 0] = [r, -r, -r]
            self.true_desired_state[:3, 1] = [-r, -r, -r]
            self.true_desired_state[:3, 2] = [-r, r, -r]
            self.true_desired_state[:3, 3] = [r, r, -r]

            self.true_desired_state[:3, 4] = [r, r, r]
            self.true_desired_state[:3, 5] = [-r, r, r]
            self.true_desired_state[:3, 6] = [-r, -r, r]
            self.true_desired_state[:3, 7] = [r, -r, r]

            self.true_desired_state[:3, 8] = [r, -r, -r]

            self.desired_state = self.true_desired_state
            self.times = [30] * self.desired_state.shape[1]
        if challenge_id == SpaceEnvironments.CUBE_ROTATE:  # floating square
            self.true_desired_state = np.zeros((12, 18))
            r = 0.075
            self.start_state[3, 0] = -3 * np.pi / 4
            self.start_state[4, 0] = 0.955

            self.true_desired_state[3:6, 0] = self.start_state[3:6, 0]
            self.true_desired_state[:3, 0] = [r, -r, -r]
            self.true_desired_state[:3, 1] = self.true_desired_state[:3, 0]
            self.true_desired_state[4, 1] = np.pi

            self.true_desired_state[3:6, 2] = self.true_desired_state[3:6, 1]
            self.true_desired_state[:3, 2] = [-r, -r, -r]
            self.true_desired_state[:3, 3] = self.true_desired_state[:3, 2]
            self.true_desired_state[4, 3] = np.pi
            self.true_desired_state[5, 3] = -np.pi / 2

            self.true_desired_state[3:6, 4] = self.true_desired_state[3:6, 3]
            self.true_desired_state[:3, 4] = [-r, r, -r]
            self.true_desired_state[:3, 5] = self.true_desired_state[:3, 4]
            self.true_desired_state[4, 5] = np.pi
            self.true_desired_state[5, 5] = -np.pi

            self.true_desired_state[3:6, 6] = self.true_desired_state[3:6, 5]
            self.true_desired_state[:3, 6] = [r, r, -r]
            self.true_desired_state[:3, 7] = self.true_desired_state[:3, 6]
            self.true_desired_state[4, 7] = 3 * np.pi / 2
            self.true_desired_state[5, 7] = -np.pi

            self.true_desired_state[3:6, 8] = self.true_desired_state[3:6, 7]
            self.true_desired_state[:3, 8] = [r, r, r]
            self.true_desired_state[:3, 9] = self.true_desired_state[:3, 8]
            self.true_desired_state[4, 9] = 2 * np.pi
            self.true_desired_state[5, 9] = -np.pi

            self.true_desired_state[3:6, 10] = self.true_desired_state[3:6, 9]
            self.true_desired_state[:3, 10] = [-r, r, r]
            self.true_desired_state[:3, 11] = self.true_desired_state[:3, 10]
            self.true_desired_state[4, 11] = 2 * np.pi
            self.true_desired_state[5, 11] = -3 * np.pi / 2

            self.true_desired_state[3:6, 12] = self.true_desired_state[3:6, 11]
            self.true_desired_state[:3, 12] = [-r, -r, r]
            self.true_desired_state[:3, 13] = self.true_desired_state[:3, 12]
            self.true_desired_state[4, 13] = 2 * np.pi
            self.true_desired_state[5, 13] = -2 * np.pi

            self.true_desired_state[3:6, 14] = self.true_desired_state[3:6, 13]
            self.true_desired_state[:3, 14] = [r, -r, r]
            self.true_desired_state[:3, 15] = self.true_desired_state[:3, 14]
            self.true_desired_state[4, 15] = 3 * np.pi / 2
            self.true_desired_state[5, 15] = -2 * np.pi

            self.true_desired_state[3:6, 16] = self.true_desired_state[3:6, 15]
            self.true_desired_state[:3, 16] = [r, -r, -r]
            self.true_desired_state[:3, 17] = self.true_desired_state[:3, 16]
            self.true_desired_state[4, 17] = 3 * np.pi / 2
            self.true_desired_state[5, 17] = -2.5 * np.pi

            self.desired_state = self.true_desired_state
            self.times = [300] * self.desired_state.shape[1]

        if challenge_id == SpaceEnvironments.SPIN:  # individual axis
            self.magnets = [self.magnets[i] for i in [4, 5]]
            self.true_desired_state = np.zeros((12, 2))

            self.true_desired_state[5, 0] = np.pi / 2

            self.desired_state = self.true_desired_state
            self.times = [360] * self.desired_state.shape[1]

        if challenge_id == SpaceEnvironments.SPIN2:  # individual axis
            self.magnets = [self.magnets[i] for i in [4, 5]]
            self.true_desired_state = np.zeros((12, 6))

            self.true_desired_state[5, 0] = np.pi / 2
            self.true_desired_state[3, 2] = np.pi / 2
            self.true_desired_state[4, 4] = np.pi / 2

            self.desired_state = self.true_desired_state
            self.times = [360] * self.desired_state.shape[1]

        if challenge_id == SpaceEnvironments.BACKANDFORTH:  # individual axis
            self.magnets = [self.magnets[i] for i in [0, 1]]
            self.magnets[0][1] += 1e-15
            self.magnets[1][1] += 1e-15

            self.true_desired_state = np.zeros((12, 8))

            self.true_desired_state[0, 0] = 0.025
            self.true_desired_state[1, 0] = 0.025
            self.true_desired_state[2, 0] = 0.025

            self.true_desired_state[0, 1] = 0.025
            self.true_desired_state[1, 1] = 0.025
            self.true_desired_state[2, 1] = -0.025

            self.true_desired_state[0, 2] = 0.025
            self.true_desired_state[1, 2] = -0.025
            self.true_desired_state[2, 2] = -0.025

            self.true_desired_state[0, 3] = 0.025
            self.true_desired_state[1, 3] = -0.025
            self.true_desired_state[2, 3] = 0.025

            self.true_desired_state[0, 4] = -0.025
            self.true_desired_state[1, 4] = -0.025
            self.true_desired_state[2, 4] = 0.025

            self.true_desired_state[0, 5] = -0.025
            self.true_desired_state[1, 5] = -0.025
            self.true_desired_state[2, 5] = -0.025

            self.true_desired_state[0, 6] = -0.025
            self.true_desired_state[1, 6] = 0.025
            self.true_desired_state[2, 6] = -0.025

            self.true_desired_state[0, 7] = -0.025
            self.true_desired_state[1, 7] = 0.025
            self.true_desired_state[2, 7] = 0.025

            self.desired_state = self.true_desired_state
            self.times = [100] * self.desired_state.shape[1]
            self.start_state[1] = 1e-15

    def spiral_helper(self, t):
        xyz = np.zeros((12))
        scale = 0.02
        rotation_scale = 2

        velocity_scale = 0.002

        xyz[0] = scale * t * np.cos(t * rotation_scale)
        xyz[1] = scale * t * np.sin(t * rotation_scale)
        xyz[2] = scale * (t - np.pi)

        xyz[6] = (
            velocity_scale
            * scale
            * (
                np.cos(t * rotation_scale)
                - velocity_scale * rotation_scale * t * np.sin(rotation_scale * t)
            )
        )
        xyz[7] = (
            velocity_scale
            * scale
            * (
                np.sin(t * rotation_scale)
                + velocity_scale * rotation_scale * t * np.cos(rotation_scale * t)
            )
        )
        xyz[8] = velocity_scale * scale

        return xyz

    def problem_specific_modification(self, state, wrench):
        pass
