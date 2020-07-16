# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:02:00 2020

@author: tabor
"""

import dataloader
import sys
sys.path.append("..") # Adds higher directory to python modules path.
dataloader.load("data/real_2020-07-05 18_59_54.452493",globals())
import visualization
import matplotlib.pyplot as plt



import numpy as np


modified_controls = np.zeros((i,4))
for j in range(i):
    modified_controls[j,controls[j][0]] = controls[j][2]
npdesiredStates = np.array(desiredStates)
npStates = np.array(states)
Plots = {'Velocity (m/s)':np.transpose(npStates[:,6:8,0]),
         'Desired velocity (m/s) ':np.transpose(npdesiredStates[:,6:8,0]),
         'Velocity error (m/s)':np.transpose(npdesiredStates[:,6:8,0])-np.transpose(npStates[:,6:8,0])}

visualization.plotStuff(Plots,'Controller')

Plots = {'Position (m)':np.transpose(npStates[:,:3,0]),
         'Desired position (m)':np.transpose(npdesiredStates[:,:3,0]),
         'Position error (m)':np.transpose(npdesiredStates[:,:3,0])-np.transpose(npStates[:,:3,0])}
visualization.plotStuff(Plots,'Controller')

Plots = {'Orientation (degrees)':np.transpose(npStates[:,5:6,0])*(180/np.pi),
         'Control (Am^2)': np.transpose(modified_controls[:,:])}
visualization.plotStuff(Plots,'Controller')


fig = plt.figure()
fig.suptitle('Path (m)')
plt.plot(npStates[:,0,0],npStates[:,1,0])
plt.scatter(npdesiredStates[:,0,0],npdesiredStates[:,1,0])
plt.show()
visualization.visualize2DPath(env,np.transpose(npStates[:,:,0]),'Controller Path')