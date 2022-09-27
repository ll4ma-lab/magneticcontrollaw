#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 17:23:35 2021

@author: tabor
"""
import numpy as np
import torch
import time
from dataclasses import dataclass
        
@dataclass 
class CylinderControlCommand:
    mag_num: int = 0
    model_num: int = 0
    dipole_strength: float = 0.0
    theta: float = 0
    seconds: float = 1.0

@dataclass 
class SphereControlCommand:
    mag_num: int = 0
    dipole_strength: float = 0.0
    psi: float = 0
    xi: float = 0
    seconds: float = 1.0


#cylinder model helper
def getOmniFrame(pos):
    Z = np.array([0, 0, 1])
    dist = np.linalg.norm(pos)
    
    omni_Z = -1*(pos/dist)
    omni_Y = np.cross(omni_Z,Z)
    omni_Y = omni_Y/np.sqrt(np.dot(omni_Y,omni_Y)) #unit vectorise
    omni_X = np.cross(omni_Y,omni_Z)
    # Combine to build rotation matrix
    omniFrame = np.transpose(np.array([omni_X, omni_Y, omni_Z]))
    return omniFrame

def averageDict(a,b):
    newDict = {}
    for key in a.keys():
        newDict[key] = ((np.array(a[key])+np.array(b[key]))/2).tolist()
    return newDict

#cylinder model helper
def rotZ(yaw,library = np):
    yaw = yaw*np.pi/180.0
    one,zero = (1.0,0.0) if library is np else (torch.tensor([1.0]),torch.tensor([0.0]))
    
    RZ = library.stack([
                library.stack([library.cos(yaw), -library.sin(yaw), zero]),
                library.stack([library.sin(yaw), library.cos(yaw), zero]),
                library.stack([zero, zero, one])]).reshape(3,3)
    return RZ

#sphere model helper
def constructOmega(angles, library = np):
    angles = angles*np.pi/180.0
    omega = library.stack([
        library.sin(angles[0])*library.cos(angles[1]),
        library.sin(angles[0])*library.sin(angles[1]),
        library.cos(angles[0])]).reshape(3,-1)
    return omega



def vectorNorm(vector,library):
    if library is np:
        norm = library.linalg.norm(vector,axis=1)#library agnostic L2 norm
    else:
        norm = library.linalg.norm(vector,dim=1)#library agnostic L2 norm
    return norm
#sphere model helper
def computeThetaAndFrame(omega,rho,library = np):
    
    omega = library.swapaxes(omega,0,1)
    rho = library.swapaxes(rho,0,1) #libraries dont like 3,n and prefer n,3
    
    cross = library.cross(omega,rho)
    
    l2Cross = vectorNorm(cross,library)

    dot = library.sum(omega*rho,1)
    if library is np:
        theta = library.arctan2(l2Cross,dot)
    else:
        theta = library.atan2(l2Cross,dot)
    omni_Z =  (rho.T/vectorNorm(rho,library)).T
    omni_Y = (cross.T/l2Cross).T
    
    omni_X = library.cross(omni_Y,omni_Z);
    # Combine to build rotation matri
    stacked = library.stack([omni_X,omni_Y,omni_Z])
    #first move the batch dimension to the end, then tranpose the rotation matrix
    omniFrame=library.swapaxes(stacked,0,1)
    omniFrame=library.swapaxes(omniFrame,1,2)


    return theta,omniFrame


#sphere model helper
def modelCallRotatedWrench(dipoleStrength,angles,modelVals,rho,library=np,zero = np.array([0])):
    frho0,trho0,frho90,fphi90,ttheta90 = modelVals
    omega = constructOmega(angles,library)
    theta,rotation = computeThetaAndFrame(omega, rho,library)
    unitStrengthWrench = library.stack([zero,
                   fphi90*library.sin(theta),
                   (-(frho90-frho0)/2)*library.cos(2*theta) + (frho90+frho0)/2,
                   ttheta90*library.sin(theta),
                   zero,
                   trho0*library.cos(theta)])
    #switch batch dim to 0
    unitStrengthWrench = library.swapaxes(unitStrengthWrench,0,1)
    if(library is np):
        unitStrengthWrench = np.expand_dims(unitStrengthWrench, 2)
        selectedWrench = [dipoleStrength[i]**2 * np.concatenate((rotation[i] @ unitStrengthWrench[i,:3],rotation[i] @ unitStrengthWrench[i,3:])) 
                          for i in range(unitStrengthWrench.shape[0])] 
        selectedWrench = np.array(selectedWrench)
    else:
        unitStrengthWrench = torch.unsqueeze(unitStrengthWrench, 2)
        
        if dipoleStrength.shape[0] == 1:
            selectedWrench = torch.cat((rotation[0]@unitStrengthWrench[0,:3],rotation[0]@ unitStrengthWrench[0,3:]),0).T
            selectedWrench = torch.unsqueeze(selectedWrench * dipoleStrength**2,2)
        else:            
            forces =torch.bmm(rotation,unitStrengthWrench[:,:3])
            torques = torch.bmm(rotation, unitStrengthWrench[:,3:])
            selectedWrench = torch.bmm(torch.cat((forces,torques),1),dipoleStrength.reshape((-1,1,1))**2)
            
    return selectedWrench  
    
class multiMagModel():
    def __init__(self,magnets,object_properties,useSphere= False):
        self.time = 0
        self.magnets = magnets
        self.object_properties = object_properties
        self.cylinderModel = cylinderModel(object_properties)
        self.sphereModel = sphereModel(object_properties)
        self.useSphere = useSphere
    def forwardPass(self,position,control):
        if(self.useSphere):
            return self.forwardSphere(position, control)
        else: 
            return self.forwardCylinder(position, control)
        
    def forwardCylinder(self,position,control):
        s = time.time()
        chosenMagnetPose = (np.array(self.magnets) - np.transpose(position)).tolist()[control.mag_num]
        omniFrame = getOmniFrame(chosenMagnetPose)
        
        if(control.model_num == 2):
            rotation = np.dot(omniFrame,rotZ(control.theta))
        else:
            rotation = omniFrame
        
        doubleRotMatrix = np.zeros((6,6))
        doubleRotMatrix[:3,:3] = rotation
        doubleRotMatrix[3:,3:] = rotation
        
        wrench = self.cylinderModel.modelCall(np.linalg.norm(chosenMagnetPose),control.dipole_strength)[control.model_num]
        
        rotatedWrench = doubleRotMatrix@wrench
        self.time += time.time()-s
        return rotatedWrench
    def forwardSphere(self,position,control):
        s = time.time()
        chosenMagnetPose = (np.array(self.magnets)[control.mag_num] - np.transpose(position)).transpose()
        dist = np.linalg.norm(chosenMagnetPose)
        rho = (-1*chosenMagnetPose).reshape((3,1))

        modelVals = self.sphereModel.modelCall(dist,1)
        wrench = modelCallRotatedWrench(np.array(control.dipole_strength).reshape(1),np.array((control.psi,control.xi)),modelVals,rho).reshape((6,1))

        self.time += time.time()-s
        return wrench
 
class sphereModel():
    def __init__(self,object_properties):
        self.cylinderModel = cylinderModel(object_properties)
        
    def modelCall(self,dist,dipoleStrength,radii=None,sigma=None):
        frho0 = self.cylinderModel.modelEquation(self.cylinderModel.constants['fzz'],dist,dipoleStrength,4,radii,sigma)
        trho0 = self.cylinderModel.modelEquation(self.cylinderModel.constants['tzz'],dist,dipoleStrength,3,radii,sigma)
        
        frho90 = self.cylinderModel.modelEquation(self.cylinderModel.constants['frhorho'],dist,dipoleStrength,4,radii,sigma)
        fphi90 = self.cylinderModel.modelEquation(self.cylinderModel.constants['frhophi'],dist,dipoleStrength,4,radii,sigma)
        ttheta90= self.cylinderModel.modelEquation(self.cylinderModel.constants['trhoz'],dist,dipoleStrength,3,radii,sigma)
        return frho0,trho0,frho90,fphi90,ttheta90
    
    
                

class cylinderModel():
    def __init__(self,object_properties):
        self.time = 0.0

        self.object_properties = object_properties
        self.freq = 15
        self.mu = 4*np.pi*1e-7
        
        self.feaConstants = {}
        self.feaConstants['fzz'] = [430,2.95,-0.101,-9.26,7]
        self.feaConstants['tzz'] = [6840,3.00,-0.0986,-13.2,6]
        self.feaConstants['frhorho'] = [266,2.60,-0.101,-7.65,7]
        self.feaConstants['frhophi'] = [6040,3.45,-0.102,-14.3,7]
        self.feaConstants['trhoz'] = [8100,3.60,-0.0985,-15.7,6]
        
        
        self.expConstants = {}
        self.expConstants['fzz'] = [467,2.81,-0.0969,-9.75,7]
        self.expConstants['tzz'] = [6900,3.35,-0.0990,-14.9,6]
        self.expConstants['frhorho'] = [282,3.20,-0.0980,-9.41,7]
        self.expConstants['frhophi'] = [5870,3.49,-0.0973,-14.6,7]
        self.expConstants['trhoz'] = [8000,3.40,-0.0928,-15.0,6]
        
        self.constants = averageDict(self.feaConstants,self.expConstants)

    def modelEquation(self,constants,distance,dipoleStrength,p0RPower,radii=None,sigma=None):
        if(radii is None):
            radii = self.object_properties.radius
        if(sigma is None):
            sigma = self.object_properties.sigma
        #P0 has r^4 for forces and r^3 for torques
        P1 = sigma*self.mu*self.freq*radii*radii
        P2 = distance/radii

        numeratorBase = constants[0]*P1
        numeratorPower = constants[1]*(P1**constants[2])
        numerator = (numeratorBase**numeratorPower) * (10**constants[3])
        
        denominator = P2**constants[4]
        P0 = numerator/denominator

        c=  self.mu*(dipoleStrength**2)*(radii**-p0RPower)

        wrench = P0*c
        
        return wrench
        
    def modelCall(self,dist,dipoleStrength):
        fzz = self.modelEquation(self.constants['fzz'],dist,dipoleStrength,4)
        tzz = self.modelEquation(self.constants['tzz'],dist,dipoleStrength,3)
        
        frhorho = self.modelEquation(self.constants['frhorho'],dist,dipoleStrength,4)
        frhophi = self.modelEquation(self.constants['frhophi'],dist,dipoleStrength,4)
        trhoz = self.modelEquation(self.constants['trhoz'],dist,dipoleStrength,3)
        
        pointingModel = np.array([0,0,fzz,0,0,tzz]).reshape((6,1))
        negPointingModel = np.array([0,0,fzz,0,0,-tzz]).reshape((6,1))
        
        rhoModel = np.array([0,-frhophi,frhorho,-trhoz,0,0]).reshape((6,1))
        return pointingModel,negPointingModel,rhoModel



