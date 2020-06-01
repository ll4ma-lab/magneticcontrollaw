# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:48:07 2020

@author: tabor
"""

import numpy as np
import time
class SplineProfile:
    def __init__(self,env,totalTimeSteps,times):
        startState = env.startState
        points = env.desiredState
        self.points = np.concatenate((startState,points),1)
        self.totalTimeSteps = totalTimeSteps
        self.time = 0.0
        self.times = times
        self.switchingTimes = [np.sum([times[i] for i in range(j)]) for j in range(len(times)+1)]

    def getDesiredState(self,step):
        start = time.time()
        stateIndex = 0
        while self.switchingTimes[stateIndex+1] <= step:
            stateIndex +=1

        lastPose = np.reshape(self.points[:,stateIndex],(-1,1))
        nextPose = np.reshape(self.points[:,stateIndex+1],(-1,1))
        x = lastPose[:6]
        xd = lastPose[6:]
        y = nextPose[:6]
        yd = nextPose[6:]
        
        As = x
        Bs = xd
        Cs = 3*y - yd -2 * xd -3*x
        Ds = xd + 2*x - yd - (y-yd)*2
        t = (step -  self.switchingTimes[stateIndex])/float(self.times[stateIndex])
        spline = np.concatenate((Ds * (t**3) + Cs*(t**2) + Bs*t + As,
                               (3*Ds * (t**2) + 2*Cs*(t) + Bs) /float(self.times[stateIndex])))
        self.time += time.time() - start
        return spline