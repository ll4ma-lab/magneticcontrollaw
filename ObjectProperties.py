#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 19:06:32 2022

@author: tabor
"""
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class ObjectProperties:
    mass: float = 1
    MOI: np.ndarray = np.eye(3)
    radius: float = 0
    sigma: float = 0
    
    def __add__(self,o):
        mass = self.mass + o.mass
        MOI = self.MOI + o.MOI
        radius = self.radius + o.radius
        sigma = self.sigma + o.sigma
        return ObjectProperties(mass=mass,MOI=MOI,radius=radius,sigma=sigma)
    def torch(self):
        return ObjectProperties(mass=torch.tensor(self.mass),
                                MOI=torch.tensor(self.MOI),
                                radius=torch.tensor(self.radius),
                                sigma=torch.tensor(self.sigma))

def buildMassMatrix(object_properties,library=np,torqueTop=True):
    mass = object_properties.mass
    MOI = object_properties.MOI

    massComponent = library.eye(3) *mass
    offDiagonal = library.zeros((3,3))
    
    if torqueTop:
        top = library.vstack((MOI,offDiagonal))    
        bottom = library.vstack((offDiagonal,massComponent))
    else:
        top = library.vstack((massComponent,offDiagonal))    
        bottom = library.vstack((offDiagonal,MOI))
    full = library.vstack((top.T,bottom.T)).T
    return full

def sphereProperties(density,radius,library=np):
    if torch.is_tensor(density) and library == np:
        density = density.numpy()
    volume = 4.0/3 * np.pi*(radius**3)
    mass = volume * density
    I = library.eye(3)*(2.0/5) *mass * (radius**2)
    return mass, I
def boatProperties(cylinder_mass,radius,cylinder_height,base_mass):
    base_x = (1/4) * base_mass *radius*radius
    base_y = base_x
    base_z = (1/2)*base_mass * radius * radius
    
    cylinder_x = (cylinder_mass/12) * (3*(radius**2+radius**2) + cylinder_height**2)
    cylinder_y = cylinder_x
    cylinder_z = (cylinder_mass/2) * (radius**2+radius**2)
    
    bases = 2
    cylinders = 1.5
    mass = base_mass * bases + cylinders* cylinder_mass
    MOI = (base_x*bases + cylinder_x*cylinders,
           base_y*bases + cylinder_y*cylinders,
           base_z*bases + cylinder_z*cylinders)
    MOI = np.diag(MOI)
    return mass,MOI
    