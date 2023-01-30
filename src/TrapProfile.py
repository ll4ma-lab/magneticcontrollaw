# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:04:51 2020

@author: tabor
"""

import numpy as np
import time


class TrapProfile:
    def __init__(self, env, times):
        start_state = env.startState
        points = env.desiredState
        self.points = np.concatenate((start_state, points), 1)
        self.time = 0.0
        self.times = times
        self.switching_times = [
            int(np.sum([times[i] for i in range(j)])) for j in range(len(times) + 1)
        ]
        print("init")

        for i in range(len(times)):
            start = self.points[:6, i]
            end = self.points[:6, i + 1]
            traj_time = self.times[i]
            self.maxVel = np.array([0.002] * 3 + [0.2] * 3)
            self.maxA = np.array([0.00003] * 3 + [0.0001] * 3)
            distance = end - start

            max_accel_distance = 2 * self.maxA * (self.maxVel / self.maxA) ** 2
            max_straight_distance = self.maxVel * (
                traj_time - 2 * self.maxVel / self.maxA
            )

            if np.any(
                np.abs(max_accel_distance + max_straight_distance) < np.abs(distance)
            ):
                print(i, " not possible")

    def get_desired_state(self, step, state):
        start_wall_time = time.time()
        state_index = 0
        # print(step)
        while self.switching_times[state_index + 1] < step:
            state_index += 1
        if self.switching_times[state_index + 1] == step:
            # self.points[:,stateIndex+1] = np.copy(state[:,0])
            print("updated")
        # print(self.switchingTimes[stateIndex])
        last_pose = np.reshape(self.points[:6, state_index], (1, -1))
        next_pose = np.reshape(self.points[:6, state_index + 1], (1, -1))

        distance = next_pose - last_pose
        traj_time = self.times[state_index]
        accel_time = (
            traj_time
            - np.sqrt(traj_time * traj_time - 4 * np.abs(distance) / self.maxA)
        ) / 2
        start = last_pose
        t = np.ones((1, 6)) * (step - self.switching_times[state_index])

        acceleration = self.maxA * np.sign(distance)

        start_vel_equation = acceleration * t
        start_pos_equation = start + acceleration * t * t / 2

        mid_vel_equation = accel_time * acceleration
        mid_pos_equation = (
            start
            + acceleration * accel_time * accel_time / 2
            + (t - accel_time) * mid_vel_equation
        )

        mid_end_position = (
            start
            + acceleration * accel_time * accel_time / 2
            + (traj_time - 2 * accel_time) * mid_vel_equation
        )

        final_vel_equation = (
            accel_time * acceleration - (accel_time - (traj_time - t)) * acceleration
        )
        final_pos_equation = (
            mid_end_position
            + (accel_time + (t - traj_time)) * mid_vel_equation
            - (acceleration * (accel_time + (t - traj_time)) ** 2) / 2
        )

        state = np.zeros((12, 1))
        for i in range(6):
            if t[0, i] < accel_time[0, i]:
                state[i, 0] = start_pos_equation[0, i]
                state[i + 6, 0] = start_vel_equation[0, i]
            elif traj_time - t[0, i] < accel_time[0, i]:
                state[i, 0] = final_pos_equation[0, i]
                state[i + 6, 0] = final_vel_equation[0, i]
            else:
                state[i, 0] = mid_pos_equation[0, i]
                state[i + 6, 0] = mid_vel_equation[0, i]

        self.time += time.time() - start_wall_time
        return state
