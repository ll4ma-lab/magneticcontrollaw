# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:54:42 2020

@author: tabor
"""

from oct2py import octave
import numpy as np
import matplotlib.pyplot as plt
def getOmniFrame(pos):
    dist = np.linalg.norm(pos)

    omni_X = -1*(pos/dist);
    
    # Set default Z vector as upward
    Z = np.array([0, 0, 1])
    #
    # Determine if default Z and X are the same
    if np.all(abs(omni_X) == abs(Z)):
        #         fprintf('X and Z are the same vectors, new perpendicular Z chosen [0 1 0]\n');
        #print('X and Z are the same vectors, new perpendicular Z chosen [0 1 0]')
        Z = np.array([0, 1, 0])#set Z to be something else if already equal
    
    # Determine Z component perpendicular to Y axis
    if np.dot(omni_X,Z) == 0:
        #         fprintf('Z and X are already perpendicular\n');
        #print('Z and X are already perpendicular')
        omni_Z = Z;
    else:
        omni_Z = np.cross(omni_X,Z)
        omni_Z = omni_Z/np.sqrt(np.dot(omni_Z,omni_Z))
            
    
    # Obtain Y axis by crossing Z and X
    omni_Y = np.cross(omni_Z,omni_X);
    # Combine to build rotation matrix
    omniFrame = np.transpose(np.array([omni_X, omni_Y, omni_Z]));
    return omniFrame




diamter = 0.05
freq = 10.0
power = 200.0
for i in range(10):
    pos = (np.random.random((3)) - 0.5) * 0.3
    
    cycle,cart = octave.eddyFTRotated(np.linalg.norm(pos),power,freq,diamter/2.0,nout=2)
    
    desired = np.random.random((6,1)) - 0.5
    
    
    costs=[]
    thetas = range(-360,360)
    for theta in thetas:
        thetarad = np.deg2rad(theta)
        Rx=np.array([[1,0,0],
                  [0,np.cos(thetarad) , -np.sin(thetarad)],
                  [0,np.sin(thetarad) , np.cos(thetarad)]])
        R = np.dot(getOmniFrame(pos),Rx)
        
        rotatedVector = np.concatenate([np.dot(R,cart[:3,0]),np.dot(R,cart[3:,0])])
        cost = (np.linalg.norm(desired-rotatedVector))
        costs.append(cost)
    plt.plot(thetas,np.array(costs) - np.min(costs))
plt.yscale("log")
plt.show()