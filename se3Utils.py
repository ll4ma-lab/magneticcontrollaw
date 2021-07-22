# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:48:58 2021

@author: Griffin Tabor
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

def buildSkew(vector):
    vector = np.reshape(vector,(3))
    skew = np.array([[0,-vector[2],vector[1]],
                       [vector[2],0,-vector[0]],
                       [-vector[1],vector[0],0]])
    return skew

def matrixLog(Ae):
    if(np.abs(np.trace(Ae)-1)>2.0):
        Ae = np.copy(Ae)
        Ae[:,0] /= np.linalg.norm(Ae[:,0]) #unit vectorize because numerical issues
        Ae[:,1] /= np.linalg.norm(Ae[:,1])
        Ae[:,2] /= np.linalg.norm(Ae[:,2])
    phi = np.arccos(0.5*(np.trace(Ae)-1))
    skew = (1/(2*np.sin(phi)))*(Ae-np.transpose(Ae))
    return skew,phi
def matrixExponential(w_hat_skew,theta):
    Rot = np.eye(3) + np.sin(theta) * w_hat_skew + (1-np.cos(theta))*np.dot(w_hat_skew,w_hat_skew)
    return Rot
def repairRotMatrix(transform):
    skew,phi = matrixLog(transform[:3,:3])
    vec = skewToVector(skew)
    skew /= np.linalg.norm(vec)
    rot = matrixExponential(skew, phi)
    newTransform = np.copy(transform)
    newTransform[:3,:3] = rot[:3,:3]
    if(np.any(np.isnan(newTransform))):
        #usually means no reasonable twist
        return transform
    return newTransform

def skewToVector(skew):
    w = np.zeros((3,1))
    w[0,0] = skew[2,1]
    w[1,0] = skew[0,2]
    w[2,0] = skew[1,0]
    return w

def buildTF(position,angles):
    worldToBodyRotation = R.from_euler('xyz',angles)
    worldToBodyR = worldToBodyRotation.as_matrix()
    
    bodyToWorldR = np.transpose(worldToBodyR)
    bodyToWorld = np.eye(4)
    bodyToWorld[:3,:3] = bodyToWorldR
    bodyToWorld[:3,3] = position

    return bodyToWorld