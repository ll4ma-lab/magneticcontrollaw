# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:48:07 2020

@author: tabor
"""

import numpy as np
import time


class SplineProfile:
    def __init__(self, env, times):
        start_state = env.start_state
        points = env.desired_state
        self.points = np.concatenate((start_state, points), 1)
        self.time = 0.0
        self.times = times
        self.switching_times = [
            int(np.sum([times[i] for i in range(j)])) for j in range(len(times) + 1)
        ]

    def get_desired_state(self, step, state):
        start = time.time()
        state_index = 0
        # print(step)
        while self.switching_times[state_index + 1] < step:
            state_index += 1
        if self.switching_times[state_index + 1] + 1 == step:
            # self.points[:,stateIndex+1] = np.copy(state[:,0])
            print("updated")
        # print(self.switchingTimes[stateIndex])
        last_pose = np.reshape(self.points[:, state_index], (-1, 1))
        # print(lastPose)
        next_pose = np.reshape(self.points[:, state_index + 1], (-1, 1))
        x = last_pose[:6]
        xd = last_pose[6:] * float(self.times[state_index])
        y = next_pose[:6]
        yd = next_pose[6:] * float(self.times[state_index])

        constant_term = x
        linear_term = xd
        quadratic_term = 3 * y - yd - 2 * xd - 3 * x
        cubic_term = xd + 2 * x - yd - (y - yd) * 2

        t = (step - self.switching_times[state_index]) / float(self.times[state_index])
        spline = np.concatenate(
            (
                cubic_term * (t**3) + quadratic_term * (t**2) + linear_term * t + constant_term,
                (3 * cubic_term * (t**2) + 2 * quadratic_term * (t) + linear_term)
                / float(self.times[state_index]),
            )
        )
        self.time += time.time() - start
        return spline
