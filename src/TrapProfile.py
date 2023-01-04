# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:04:51 2020

@author: tabor
"""

import numpy as np
import time


class TrapProfile:
    def __init__(self, env, totalTimeSteps, times):
        startState = env.startState
        points = env.desiredState
        self.points = np.concatenate((startState, points), 1)
        self.totalTimeSteps = totalTimeSteps
        self.time = 0.0
        self.times = times
        self.switchingTimes = [
            int(np.sum([times[i] for i in range(j)])) for j in range(len(times) + 1)
        ]
        print("init")

        for i in range(len(times)):
            start = self.points[:6, i]
            end = self.points[:6, i + 1]
            trajTime = self.times[i]
            self.maxVel = np.array([0.002] * 3 + [0.2] * 3)
            self.maxA = np.array([0.00003] * 3 + [0.0001] * 3)
            distance = end - start

            maxAccelDistance = 2 * self.maxA * (self.maxVel / self.maxA) ** 2
            maxStraightDistance = self.maxVel * (trajTime - 2 * self.maxVel / self.maxA)

            if np.any(
                np.abs(maxAccelDistance + maxStraightDistance) < np.abs(distance)
            ):
                print(i, " not possible")

    def getDesiredState(self, step, state):
        startWallTime = time.time()
        stateIndex = 0
        # print(step)
        while self.switchingTimes[stateIndex + 1] < step:
            stateIndex += 1
        if self.switchingTimes[stateIndex + 1] == step:
            # self.points[:,stateIndex+1] = np.copy(state[:,0])
            print("updated")
        # print(self.switchingTimes[stateIndex])
        lastPose = np.reshape(self.points[:6, stateIndex], (1, -1))
        nextPose = np.reshape(self.points[:6, stateIndex + 1], (1, -1))

        distance = nextPose - lastPose
        trajTime = self.times[stateIndex]
        accelTime = (
            trajTime - np.sqrt(trajTime * trajTime - 4 * np.abs(distance) / self.maxA)
        ) / 2
        start = lastPose
        t = np.ones((1, 6)) * (step - self.switchingTimes[stateIndex])

        acceleration = self.maxA * np.sign(distance)

        startVelEquation = acceleration * t
        startPosEquation = start + acceleration * t * t / 2

        midVelEquation = accelTime * acceleration
        midPosEquation = (
            start
            + acceleration * accelTime * accelTime / 2
            + (t - accelTime) * midVelEquation
        )

        midEndPosition = (
            start
            + acceleration * accelTime * accelTime / 2
            + (trajTime - 2 * accelTime) * midVelEquation
        )

        finalVelEquation = (
            accelTime * acceleration - (accelTime - (trajTime - t)) * acceleration
        )
        finalPosEquation = (
            midEndPosition
            + (accelTime + (t - trajTime)) * midVelEquation
            - (acceleration * (accelTime + (t - trajTime)) ** 2) / 2
        )

        state = np.zeros((12, 1))
        for i in range(6):
            if t[0, i] < accelTime[0, i]:
                state[i, 0] = startPosEquation[0, i]
                state[i + 6, 0] = startVelEquation[0, i]
            elif trajTime - t[0, i] < accelTime[0, i]:
                state[i, 0] = finalPosEquation[0, i]
                state[i + 6, 0] = finalVelEquation[0, i]
            else:
                state[i, 0] = midPosEquation[0, i]
                state[i + 6, 0] = midVelEquation[0, i]

        self.time += time.time() - startWallTime
        return state
