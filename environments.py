# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:20:28 2020

@author: tabor
"""
import numpy as np

class PlanarProblem:
    def __init__(self,challengeID=0,):
        self.trajectoryDimensions = 'xyt'
        width = 0.38
        length = 0.62
        height = 0.15
        
        self.lims = [(-width/2,width/2)
               ,(-length/2,length/2)
               ,(-height/2,height/2)]
        
        xOffset = -0.05
        yOffset = -0.15
        z = 0.12
        magnets = np.array([(0.12, 0.120, 0.15)
                            ,(-0.120, 0.120,0.15)
                            ,(-0.120, -0.120, 0.15)
                            ,(0.12, -0.120, 0.15)])
        self.magnets = magnets.tolist()
        

        self.startState = np.zeros((12,1))
        
        if(challengeID==0 or challengeID==1):
            self.TrueDesiredState = np.zeros((12,5))
            self.TrueDesiredState[:3,0] = [0.075,-0.15,0]
            self.TrueDesiredState[:3,1] = [-0.075,-0.15,0]
            self.TrueDesiredState[:3,2] = [-0.075,0.15,0]
            self.TrueDesiredState[:3,3] = [0.075,0.15,0]
            self.TrueDesiredState[:3,4] = [0.075,-0.15,0]

        
            self.desiredState= self.TrueDesiredState
        if(challengeID==1):
            angle = 45 /(180/np.pi) 
            R = np.array([[np.cos(angle),-np.sin(angle)],
                          [np.sin(angle),np.cos(angle)]])
            
            for i in range(5):   
                self.TrueDesiredState[:2,i] = np.dot(R,self.TrueDesiredState[:2,i])
        
            self.desiredState= self.TrueDesiredState

        if(challengeID==2):
            self.TrueDesiredState = np.zeros((12,5))
            self.TrueDesiredState[5,0] = np.pi /2
            self.TrueDesiredState[5,1] = np.pi 
            self.TrueDesiredState[5,2] = 0
            self.TrueDesiredState[5,3] = np.pi
            self.TrueDesiredState[5,4] = -np.pi

        
            self.desiredState= self.TrueDesiredState
        if(challengeID==3):
            self.TrueDesiredState = np.zeros((12,10))
            self.TrueDesiredState[:3,0] = [0.075,-0.15,0]
            self.TrueDesiredState[:3,1] = self.TrueDesiredState[:3,0]
            self.TrueDesiredState[5,1] = np.pi/2
            self.TrueDesiredState[5,2] = np.pi/2

            self.TrueDesiredState[:3,2] = [-0.075,-0.15,0]
            self.TrueDesiredState[:3,3] = self.TrueDesiredState[:3,2]
            self.TrueDesiredState[5,3] = np.pi
            self.TrueDesiredState[5,4] = np.pi


            self.TrueDesiredState[:3,4] = [-0.075,0.15,0]
            self.TrueDesiredState[:3,5] = self.TrueDesiredState[:3,4]
            self.TrueDesiredState[5,5] = 3.0 *np.pi /2.0
            self.TrueDesiredState[5,6] = 3.0 *np.pi /2.0


            self.TrueDesiredState[:3,6] = [0.075,0.15,0]
            self.TrueDesiredState[:3,7] = self.TrueDesiredState[:3,6]
            self.TrueDesiredState[5,7] = 2.0 *np.pi
            self.TrueDesiredState[5,8] = 2.0 *np.pi

            
            self.TrueDesiredState[:3,8] = [0.075,-0.15,0]
            self.TrueDesiredState[:3,9] = self.TrueDesiredState[:3,8]
            self.TrueDesiredState[5,9] = 2.0 *np.pi + np.pi/2


            

        
            self.desiredState= self.TrueDesiredState