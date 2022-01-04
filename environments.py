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
        magnets = np.array([(0.093, 0.111, 0.15) #marker 0.365m away
                            ,(-0.108, 0.101,0.15)
                            ,(-0.096, -0.098, 0.15)
                            ,(0.102, -0.091, 0.15)])
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
            self.times = [240]* self.desiredState.shape[1]
            self.times[6] = 360
            self.times[7] = 360
            
            self.times[13] = 360
            self.times[14] = 360


            #self.trajectoryDimensions = 'xy'
        import warnings
        if(len(self.trajectoryDimensions)<3):
            warnings.warn('planar environment without yaw control can have numerical issues as yaw vel gets high')
    def problem_specific_modification(self,state,wrench=None):

        # note state has TF and body frame velocity, trying to modify vel means probably rotating to world frame  
        bodyToWorld = state[0][:3,:3]
        linearVelocity = np.dot(bodyToWorld,state[2])
        rotationalVelocity = np.dot(bodyToWorld,state[1])
        
        worldToBody = np.linalg.inv(bodyToWorld)
        
        linearVelocity[2] = 0
        rotationalVelocity[0] = 0
        rotationalVelocity[1] = 0
        
        state[2][:] = np.dot(worldToBody,linearVelocity)
        state[1][:] = np.dot(worldToBody,rotationalVelocity)
        

        if(wrench is not None):
            wrench[2] = 0 #z force
            wrench[3] = 0 #roll torque
            wrench[4] = 0 #pitch torque
            

class SpaceProblem:
    def __init__(self,challengeID=0,):
        self.trajectoryDimensions = 'xyzrpt'
        width = 0.001
        length = 0.001
        height = 0.001
        self.visualizationBoundaryX = (-0.3,0.3)
        self.visualizationBoundaryY = (-0.3,0.3)
        self.visualizationBoundaryZ = (-0.3,0.3)
        self.az = -43
        self.el = -160
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
        
        magnets = np.array([(0.15, 0.00, 0)
                    ,(-0.15, 0.0,0)
                    ,(0.00, 0.150, 0.0)
                    ,(0.00, -0.150, 0.0)
                   ,(1e-10, 0.000, 0.15)
                    ,(1e-10, 0.00, -0.15)])
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
            self.times = [480]* self.desiredState.shape[1]
        
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
        if(challengeID==2): #spiral WIP
            i = 0
            self.startState[:,0] = self.spiralHelper(0) 
            components = 40
            self.TrueDesiredState = np.zeros((12,components))
            for i in range(components):
                self.TrueDesiredState[:,i] = self.spiralHelper(i*2*np.pi/components)
            self.desiredState= self.TrueDesiredState
            self.times = [120]* self.desiredState.shape[1]
        if(challengeID==3):
            self.startState[:3,0] = np.linspace(0.05,0.05,3) 
            self.startState[3:6,0] = np.linspace(0.5,1.5,3)
 
            self.TrueDesiredState = np.zeros((12,1))

            self.desiredState= self.TrueDesiredState
            self.times = [600]
        if(challengeID==4): #floating square
            self.TrueDesiredState = np.zeros((12,9))
            r = 0.075
            self.TrueDesiredState[:3,0] = [r,-r,-r]
            self.TrueDesiredState[:3,1] = [-r,-r,-r]
            self.TrueDesiredState[:3,2] = [-r,r,-r]
            self.TrueDesiredState[:3,3] = [r,r,-r]
            
            self.TrueDesiredState[:3,4] = [r,r,r]
            self.TrueDesiredState[:3,5] = [-r,r,r]
            self.TrueDesiredState[:3,6] = [-r,-r,r]
            self.TrueDesiredState[:3,7] = [r,-r,r]
            
            self.TrueDesiredState[:3,8] = [r,-r,-r]

            self.desiredState= self.TrueDesiredState
            self.times = [300]* self.desiredState.shape[1]
        if(challengeID==5): #floating square
            self.TrueDesiredState = np.zeros((12,18))
            r = 0.075
            self.startState[3,0] = -3*np.pi/4
            self.startState[4,0] = 0.955 
            
            self.TrueDesiredState[3:6,0] = self.startState[3:6,0]
            self.TrueDesiredState[:3,0] = [r,-r,-r]
            self.TrueDesiredState[:3,1] = self.TrueDesiredState[:3,0]
            self.TrueDesiredState[4,1] = np.pi 
    
            self.TrueDesiredState[3:6,2] = self.TrueDesiredState[3:6,1]
            self.TrueDesiredState[:3,2] = [-r,-r,-r]
            self.TrueDesiredState[:3,3] = self.TrueDesiredState[:3,2]
            self.TrueDesiredState[4,3] = np.pi 
            self.TrueDesiredState[5,3] = -np.pi/2 

            self.TrueDesiredState[3:6,4] = self.TrueDesiredState[3:6,3]
            self.TrueDesiredState[:3,4] = [-r,r,-r]
            self.TrueDesiredState[:3,5] = self.TrueDesiredState[:3,4]
            self.TrueDesiredState[4,5] = np.pi 
            self.TrueDesiredState[5,5] = -np.pi 
            
            self.TrueDesiredState[3:6,6] = self.TrueDesiredState[3:6,5]
            self.TrueDesiredState[:3,6] = [r,r,-r]
            self.TrueDesiredState[:3,7] = self.TrueDesiredState[:3,6]
            self.TrueDesiredState[4,7] = 3*np.pi/2 
            self.TrueDesiredState[5,7] = -np.pi 

            self.TrueDesiredState[3:6,8] = self.TrueDesiredState[3:6,7]
            self.TrueDesiredState[:3,8] = [r,r,r]
            self.TrueDesiredState[:3,9] = self.TrueDesiredState[:3,8]
            self.TrueDesiredState[4,9] = 2*np.pi 
            self.TrueDesiredState[5,9] = -np.pi 
                        
            self.TrueDesiredState[3:6,10] = self.TrueDesiredState[3:6,9]
            self.TrueDesiredState[:3,10] = [-r,r,r]
            self.TrueDesiredState[:3,11] = self.TrueDesiredState[:3,10]
            self.TrueDesiredState[4,11] = 2*np.pi
            self.TrueDesiredState[5,11] = -3*np.pi/2

            
            self.TrueDesiredState[3:6,12] = self.TrueDesiredState[3:6,11]
            self.TrueDesiredState[:3,12] = [-r,-r,r]
            self.TrueDesiredState[:3,13] = self.TrueDesiredState[:3,12]
            self.TrueDesiredState[4,13] = 2*np.pi
            self.TrueDesiredState[5,13] = -2*np.pi

            self.TrueDesiredState[3:6,14] = self.TrueDesiredState[3:6,13]
            self.TrueDesiredState[:3,14] = [r,-r,r]
            self.TrueDesiredState[:3,15] = self.TrueDesiredState[:3,14]
            self.TrueDesiredState[4,15] = 3*np.pi/2
            self.TrueDesiredState[5,15] = -2*np.pi
            
            self.TrueDesiredState[3:6,16] = self.TrueDesiredState[3:6,15]
            self.TrueDesiredState[:3,16] = [r,-r,-r]
            self.TrueDesiredState[:3,17] = self.TrueDesiredState[:3,16]
            self.TrueDesiredState[4,17] = 3*np.pi/2
            self.TrueDesiredState[5,17] = -2.5*np.pi

            self.desiredState= self.TrueDesiredState
            self.times = [300]* self.desiredState.shape[1]
        if(challengeID==6):

            self.TrueDesiredState = np.zeros((12,1))
            self.TrueDesiredState[0]=1e-6
            self.desiredState= self.TrueDesiredState
            self.times = [20]  
    def spiralHelper(self, t):
        xyz = np.zeros((12))
        scale = 0.02
        rotationScale = 2
        
        velocityScale = 0.002        
        
        xyz[0] = scale*t*np.cos(t*rotationScale)
        xyz[1] = scale*t*np.sin(t*rotationScale)
        xyz[2] = scale*(t - np.pi)
        
        xyz[6] = velocityScale*scale*(np.cos(t*rotationScale) - velocityScale*rotationScale*t*np.sin(rotationScale*t))
        xyz[7] = velocityScale*scale*(np.sin(t*rotationScale) + velocityScale*rotationScale*t*np.cos(rotationScale*t))
        xyz[8] = velocityScale*scale
        
        return xyz

        
    def problem_specific_modification(self,state,wrench):
        pass