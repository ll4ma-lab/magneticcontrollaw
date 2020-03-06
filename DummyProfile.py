# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:42:58 2020

@author: tabor
"""
import numpy as np
class DummyProfile:
    def __init__(self,startState,points,totalTimeSteps):
        self.points = points
        self.totalTimeSteps = totalTimeSteps
    def getDesiredState(self,step):
        currIndex = step/(self.totalTimeSteps/5)
        #print(currIndex)
        return np.reshape(self.points[:,currIndex],(-1,1))
