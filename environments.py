# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:20:28 2020

@author: tabor
"""
import numpy as np

class PlanarProblem:
    def __init__(self,UnitConversionFactor = 1,challengeID=0,):
        self.trajectoryDimensions = 'xyt'
        width = 0.38
        length = 0.62
        height = 0.15
        
        self.lims = [(-width/2,width/2)
               ,(-length/2,length/2)
               ,(-height/2,height/2)]
        
        yOffset = 0.075
        xOffset = -0.05
        magnets = np.array([(-width/2 -xOffset, -length/2-yOffset, 0),(width/2 +xOffset, -length/2-yOffset, 0),
                   (-width/2 -xOffset, length/2+yOffset, 0),(width/2 +xOffset, +length/2+yOffset, 0)])
        self.magnets = magnets.tolist()
        

        self.startState = np.zeros((12,1))
        
        if(challengeID==0):
            self.TrueDesiredState = np.zeros((12,1))
            self.TrueDesiredState[:3,0] = [0.075,-0.15,0]
        
            self.desiredState= self.TrueDesiredState * UnitConversionFactor
        

         
