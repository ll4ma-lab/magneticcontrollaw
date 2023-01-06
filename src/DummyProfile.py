# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:42:58 2020

@author: tabor
"""
import numpy as np


class DummyProfile:
    def __init__(self, env, total_time_steps, times):
        start_state = env.startState
        points = env.desiredState
        self.points = points
        self.total_time_steps = total_time_steps
        self.time = 0

    def get_desired_state(self, step, state):
        curr_index = (step - 1) // (self.total_time_steps)
        return np.reshape(self.points[:, curr_index], (-1, 1))
