# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:20:28 2020

@author: tabor
"""
import numpy as np

class PlanarProblem:
    def __init__(self,challengeID=0):
        self.trajectoryDimensions = 'xyt'
        width = 0.38
        length = 0.62
        height = 0.15
        self.visualizationBoundary = (-length/2,length/2)
        self.visualizationBoundaryX = (-length/2,length/2)
        self.visualizationBoundaryY = (-width/2,width/2)
        self.visualizationBoundaryZ = (-width/2,width/2)        
        self.az = -90
        self.el = -90
        self.lims = [(-width/2,width/2)
               ,(-length/2,length/2)
               ,(-height/2,height/2)]
        magnets = np.array([(0.10, 0.100, 0.165)
                            ,(-0.100, 0.100,0.165)
                            ,(-0.100, -0.100, 0.165)
                            ,(0.100, -0.100, 0.165)])
        self.magnets = magnets.tolist()
        
        self.radius = 0.065
        linearTolerance = 0.005
        degTolerance = 5.0
        rotationalTolerance =  np.pi * degTolerance/180.0
        
        self.tolerances = (linearTolerance,rotationalTolerance)
        
        self.startState = np.zeros((12,1))
        squareRadius = 0.075
        if(challengeID==-1): #line 
            self.TrueDesiredState = np.zeros((12,2))
            self.TrueDesiredState[:3,0] = [squareRadius,-squareRadius,0]
            #self.TrueDesiredState[:3,1] = [0.075,-0.075,0]
            self.TrueDesiredState[:3,1] = [-squareRadius,-squareRadius,0]
            self.desiredState= self.TrueDesiredState
            self.times = [180]* self.desiredState.shape[1]
            self.times[0] = 180
        if(challengeID==0 or challengeID==1): #square
            self.TrueDesiredState = np.zeros((12,5))
            self.TrueDesiredState[:3,0] = [squareRadius,-squareRadius,0]
            self.TrueDesiredState[:3,1] = [-squareRadius,-squareRadius,0]
            self.TrueDesiredState[:3,2] = [-squareRadius,squareRadius,0]
            self.TrueDesiredState[:3,3] = [squareRadius,squareRadius,0]
            self.TrueDesiredState[:3,4] = [squareRadius,-squareRadius,0]

        
            self.desiredState= self.TrueDesiredState
            self.times = [240]* self.desiredState.shape[1]
            self.trajectoryDimensions = 'xy'
        if(challengeID==6): #square fixed rotation
            self.TrueDesiredState = np.zeros((12,5))
            self.TrueDesiredState[:3,0] = [squareRadius,-squareRadius,0]
            self.TrueDesiredState[:3,1] = [-squareRadius,-squareRadius,0]
            self.TrueDesiredState[:3,2] = [-squareRadius,squareRadius,0]
            self.TrueDesiredState[:3,3] = [squareRadius,squareRadius,0]
            self.TrueDesiredState[:3,4] = [squareRadius,-squareRadius,0]

        
            self.desiredState= self.TrueDesiredState
            self.times = [240]* self.desiredState.shape[1]
        if(challengeID==1): #diamond
            angle = 45 /(180/np.pi) 
            R = np.array([[np.cos(angle),-np.sin(angle)],
                          [np.sin(angle),np.cos(angle)]])
            
            for i in range(5):   
                self.TrueDesiredState[:2,i] = np.dot(R,self.TrueDesiredState[:2,i])
        
            self.desiredState= self.TrueDesiredState
            self.times = [180]* self.desiredState.shape[1]
            self.trajectoryDimensions = 'xyt'

        if(challengeID==2): #stationary rotation
            self.TrueDesiredState = np.zeros((12,5))
            #self.TrueDesiredState[1,:] = -0.075

            self.TrueDesiredState[5,0] = -np.pi /4
            self.TrueDesiredState[5,1] = 0 
            self.TrueDesiredState[5,2] = np.pi/4
            self.TrueDesiredState[5,3] = np.pi/2
            self.TrueDesiredState[5,4] = 0

            #self.TrueDesiredState[4,:] = np.pi/2
        
            self.desiredState= self.TrueDesiredState
            self.times = [180]* self.desiredState.shape[1]
            self.times[0] = 360

        if(challengeID==3): # rotate while square
            self.startState[5] = -np.pi/4

            self.TrueDesiredState = np.zeros((12,10))
            self.TrueDesiredState[5,0] = -np.pi/4


            self.TrueDesiredState[:3,0] = [squareRadius,-squareRadius,0]
            self.TrueDesiredState[:3,1] = self.TrueDesiredState[:3,0]
            self.TrueDesiredState[5,1] = -np.pi
            self.TrueDesiredState[5,2] = -np.pi

            self.TrueDesiredState[:3,2] = [-squareRadius,-squareRadius,0]
            self.TrueDesiredState[:3,3] = self.TrueDesiredState[:3,2]
            self.TrueDesiredState[5,3] = -3.0 *np.pi /2.0
            self.TrueDesiredState[5,4] = -3.0 *np.pi /2.0


            self.TrueDesiredState[:3,4] = [-squareRadius,squareRadius,0]
            self.TrueDesiredState[:3,5] = self.TrueDesiredState[:3,4]
            self.TrueDesiredState[5,5] = -2.0 *np.pi
            self.TrueDesiredState[5,6] = -2.0 *np.pi


            self.TrueDesiredState[:3,6] = [squareRadius,squareRadius,0]
            self.TrueDesiredState[:3,7] = self.TrueDesiredState[:3,6]
            self.TrueDesiredState[5,7] = -2.5 *np.pi
            self.TrueDesiredState[5,8] = -2.5 *np.pi

            
            self.TrueDesiredState[:3,8] = [squareRadius,-squareRadius,0]
            self.TrueDesiredState[:3,9] = self.TrueDesiredState[:3,8]
            self.TrueDesiredState[5,9] = -3.0 *np.pi

        
            self.desiredState= self.TrueDesiredState
            self.times = [240]* self.desiredState.shape[1]
            
        if(challengeID==4): #infinity
            vel = 0.0005            
            
            self.TrueDesiredState = np.zeros((12,8))
            self.TrueDesiredState[:3,0] = [-0.06,0.04,0]
            self.TrueDesiredState[6:9,0] = [-vel,0,0]

            self.TrueDesiredState[:3,1] = [-0.12,0,0] #left
            self.TrueDesiredState[6:9,1] = [0,-vel/2,0]

            self.TrueDesiredState[:3,2] = [-0.06,-0.04,0]
            self.TrueDesiredState[6:9,2] = [vel,0,0]

            self.TrueDesiredState[:3,3] = [-0.0,0.0,0] # center


            self.TrueDesiredState[:3,4] = [0.06,0.04,0]
            self.TrueDesiredState[6:9,4] = [vel,0,0]

            self.TrueDesiredState[:3,5] = [0.12,0,0] #left
            self.TrueDesiredState[6:9,5] = [0,-vel/2,0]

            self.TrueDesiredState[:3,6] = [0.06,-0.04,0]
            self.TrueDesiredState[6:9,6] = [-vel,0,0]

            self.TrueDesiredState[:3,7] = [-0.0,0.0,0] # center
        
            self.desiredState= self.TrueDesiredState
            self.times = [180]* self.desiredState.shape[1]
            self.times[0] = 180
            #self.trajectoryDimensions = 'xy'
        if(challengeID==5): #U
            vel = 0.1       
            self.TrueDesiredState = np.zeros((12,15))
            self.TrueDesiredState[:3,0] = [5,-7,0] 
            self.TrueDesiredState[:3,1] = [3,-7,0]
            self.TrueDesiredState[:3,2] = [3,-11,0]
            self.TrueDesiredState[:3,3] = [13,-11,0]
            self.TrueDesiredState[:3,4] = [13,-7,0]
            self.TrueDesiredState[:3,5] = [11,-7,0]
            
            self.TrueDesiredState[:3,6] = [0,13,0]
            self.TrueDesiredState[6:9,6] = [-0.2,0,0]

            
            
            self.TrueDesiredState[:3,7] = [-11,-7,0]
            self.TrueDesiredState[:3,8] = [-13,-7,0]
            self.TrueDesiredState[:3,9] = [-13,-11,0]
            self.TrueDesiredState[:3,10] = [-3,-11,0]
            self.TrueDesiredState[:3,11] = [-3,-7,0]
            self.TrueDesiredState[:3,12] = [-5,-7,0] 
            
            self.TrueDesiredState[:3,13] = [-0,9,0] 
            self.TrueDesiredState[:3,14] = [5,-7,0] 
            self.TrueDesiredState[6:9,13] = [0.08,0,0]

            self.TrueDesiredState = self.TrueDesiredState * 0.005

        
            self.desiredState= self.TrueDesiredState
            self.times = [180]* self.desiredState.shape[1]
            self.times[6] = 240
            self.times[7] = 240
            
            self.times[13] = 240
            self.times[14] = 240


            #self.trajectoryDimensions = 'xy'
    def problem_specific_modification(self,state):
        #set z rp + their velocity to 0

        state[2] = 0
        state[3] = 0
        state[4] = 0
        
        state[2 + 6 ] = 0
        state[3 + 6 ] = 0
        state[4 + 6] = 0


class SpaceProblem:
    def __init__(self,challengeID=0,):
        self.trajectoryDimensions = 'xyzrpt'
        width = 0.001
        length = 0.001
        height = 0.001
        self.visualizationBoundaryX = (-0.3,0.3)
        self.visualizationBoundaryY = (-0.3,0.3)
        self.visualizationBoundaryZ = (-0.3,0.3)
        self.az = 14
        self.el = -150
        self.lims = [(-width/2,width/2)
               ,(-length/2,length/2)
               ,(-height/2,height/2)]
        
        linearTolerance = 0.005
        degTolerance = 10.0
        rotationalTolerance =  np.pi * degTolerance/180.0
        
        self.tolerances = (linearTolerance,rotationalTolerance)
        self.radius = 0.065

        num_pts = 6
        from numpy import pi, cos, sin, arccos, arange

        indices = arange(0, num_pts, dtype=float) + 0.5
        
        phi = arccos(1 - 2*indices/num_pts)
        theta = pi * (1 + 5**0.5) * indices
        
        x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi);
        magnets = np.transpose(np.array((x,y,z))) * 0.15
        
#        magnets = np.array([(0.15, 0.00, 0)
#                    ,(-0.15, 0.0,0)
#                    ,(0.00, 0.150, 0.0)
#                    ,(0.00, -0.150, 0.0)
#                    ,(0.000, 0.000, 0.15)
#                    ,(0.00, 0.00, -0.15)])
        self.magnets = magnets.tolist()
        self.startState = np.zeros((12,1))
        
        if(challengeID==0): #floating square
            self.TrueDesiredState = np.zeros((12,5))
            self.TrueDesiredState[:3,0] = [0.075,-0.075,-0.075]
            self.TrueDesiredState[:3,1] = [-0.075,-0.075,-0.075]
            self.TrueDesiredState[:3,2] = [-0.075,0.075,0.075]
            self.TrueDesiredState[:3,3] = [0.075,0.075,0.075]
            self.TrueDesiredState[:3,4] = [0.075,-0.075,0]
        
        
            self.desiredState= self.TrueDesiredState
            self.times = [240]* self.desiredState.shape[1]
        
        if(challengeID==1): # individual axis
            self.TrueDesiredState = np.zeros((12,12))
            self.TrueDesiredState[0,0] = 0.1
            
            self.TrueDesiredState[1,2] = 0.1

            self.TrueDesiredState[2,4] = 0.1

            self.TrueDesiredState[3,6] = np.pi/2
            
            self.TrueDesiredState[4,8] = np.pi/2

            self.TrueDesiredState[5,10] = np.pi/2
        
            self.desiredState= self.TrueDesiredState
            self.times = [180]* self.desiredState.shape[1]
        if(challengeID==0): #floating square
            self.TrueDesiredState = np.zeros((12,5))
            self.TrueDesiredState[:3,0] = [0.075,-0.075,-0.075]
            self.TrueDesiredState[:3,1] = [-0.075,-0.075,-0.075]
            self.TrueDesiredState[:3,2] = [-0.075,0.075,0.075]
            self.TrueDesiredState[:3,3] = [0.075,0.075,0.075]
            self.TrueDesiredState[:3,4] = [0.075,-0.075,0]
        
        
            self.desiredState= self.TrueDesiredState
            self.times = [240]* self.desiredState.shape[1]

    def problem_specific_modification(self,state):
        pass