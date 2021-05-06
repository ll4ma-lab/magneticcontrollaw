# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:40:03 2021

@author: Griffin Tabor
"""
import numpy as np
import time
from se3Utils import buildSkew,repairRotMatrix
class discreteTimeSimulator:
    def __init__(self,state,timestep,object_properties):
        (mass,MOI) = object_properties
        self.t = timestep
        self.dt = 0.002
        self.mass = mass
        self.MOI = MOI
    
        massMatrix = np.eye(3) * mass
        
        from scipy.spatial.transform import Rotation as R
        
        worldToBodyRotation = R.from_euler('xyz',state[3:6,0])
        worldToBodyR = worldToBodyRotation.as_dcm()
        
        
        bodyToWorldR = np.transpose(worldToBodyR)
        bodyToWorld = np.eye(4)
        bodyToWorld[:3,:3] = bodyToWorldR
        bodyToWorld[:3,3] = state[:3,0]


        Ωsc = np.dot(worldToBodyR,state[9:12,:])
        Vsc = np.dot(worldToBodyR,state[6:9,:])
        self.state = (bodyToWorld,Ωsc,Vsc)
        self.Isc = np.zeros((6,6))
        self.Isc[:3,:3] = MOI
        self.Isc[3:,3:] = massMatrix
        
        self.steps = int(self.t/self.dt)
        
        self.time = 0

    def simulate(self,control):
        self.state = self.forwardPass(self.state,control)
        return self.state
    def forwardPass(self,state,FT):
        start = time.time()

        #Notation and math from https://downloads.hindawi.com/journals/ijae/2017/4034328.pdf
        (bodyToWorld0,Ωsc0,Vsc0) = state
        
        bodyToWorld = np.copy(bodyToWorld0)
        Ωsc = np.copy(Ωsc0)
        Vsc = np.copy(Vsc0)
        
        FT = np.reshape(FT,(6,1))
        
        for i in range(self.steps):
            rotrot = np.concatenate((np.concatenate((np.transpose(bodyToWorld[:3,:3]),np.zeros((3,3))),1),np.concatenate((np.zeros((3,3)),np.transpose(bodyToWorld[:3,:3])),1)),0)
            TF = np.zeros(FT.shape)
            TF[:3] = FT[3:]
            TF[3:] = FT[:3]
            bodyTF = np.dot(rotrot,TF)
        
            ξsc = np.concatenate([Ωsc,Vsc])
            ΩscX = buildSkew(Ωsc)
            VscX = buildSkew(Vsc)
            ξscV = np.concatenate((np.concatenate((ΩscX,Vsc),1),np.zeros((1,4))),0)
            
            adξ = np.concatenate((np.concatenate((ΩscX,np.zeros((3,3))),1),np.concatenate((VscX,ΩscX),1)),0)
            adξs = np.transpose(adξ)
            
            
            rightSide = np.dot(adξs,np.dot(self.Isc,ξsc)) +  bodyTF
            
            ξsc_dot = np.dot(np.linalg.inv(self.Isc), rightSide)
            bodyToStrange = np.eye(4)
            bodyToStrange[:3,:3] = bodyToWorld[:3,:3]
            bodyToWorld_dot = np.dot(bodyToStrange,ξscV) #gsc_dot = gsc * ξscV
            
            bodyToWorld+= bodyToWorld_dot*self.dt
           
            
            Ωsc += ξsc_dot[:3,:] * self.dt
            Vsc += ξsc_dot[3:,:] * self.dt
        self.time += time.time() - start
        #bodyToWorld = repairRotMatrix(bodyToWorld)
        return (bodyToWorld,Ωsc,Vsc)
   # def setState(self,state):
   #     self.state = np.copy(state)
def sphereProperties(density,diameter):
    radius = diameter / 2.0
    volume = 4.0/3 * np.pi*np.power(radius,3)
    mass = volume * density
    I = np.eye(3)*(2.0/5) *mass * np.power(radius,2)
    return mass, I
