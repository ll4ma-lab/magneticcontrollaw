# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:42:58 2020

@author: tabor
"""
import numpy as np


class DummyProfile:
    def __init__(self, env, totalTimeSteps, times):
        startState = env.startState
        points = env.desiredState
        self.points = points
        self.totalTimeSteps = totalTimeSteps
        self.time = 0

    def getDesiredState(self, step, state):
        currIndex = (step - 1) // (self.totalTimeSteps)
        return np.reshape(self.points[:, currIndex], (-1, 1))
