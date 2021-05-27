# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:02:00 2020

@author: tabor
"""

import dataloader
import sys
sys.path.append("..") # Adds higher directory to python modules path.
orientation = True

if orientation:
    dataloader.load("C:/Users/Griffin Tabor/Documents/magneticLaw/data/sim_2700s_2021-05-25 14.31.31.094727",globals())
else: 
    dataloader.load("C:/Users/Griffin Tabor/Documents/magneticLaw/data/sim_2700s_2021-05-25 14.50.51.553896",globals())

import visualization
import matplotlib.pyplot as plt
import SplineProfile



import numpy as np



Plots = {'Position (m)':np.transpose(positions),
         'Velocity (m/s)':np.transpose(linearVelocity),
         'Rot velocity (_/s)':np.transpose(rotationalVelocity),
         "Force ":np.transpose(np.array(wrenches)[:,:3,0]),
         "Torque ":np.transpose(np.array(wrenches)[:,3:,0]),
         'Desired position (m)':np.transpose(npdesiredStates[:,:3,0]),
         'Desired velocity (m/s) ':np.transpose(npdesiredStates[:,6:9,0])}

visualization.plotStuff(Plots,'Controller')

Plots = {'Position Error(m)':np.abs(np.transpose(remainingError[:,:3,0])),
         'Orientation Error(radians)':-np.transpose(remainingError[:,3:6,0])}

times = list(range(numSteps))

startTF = buildTF(env.startState[:3,0],env.startState[3:6,0])

remainingError = np.array([controller.computeError(startTF,state[0]) for state in states])
positions = np.transpose(remainingError[:,:3,0]) 
angles = np.transpose(remainingError[:,3:6,0])
visualization.plotStuff(Plots,'Error')

fig, ax1 = plt.subplots()


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('Orientation (rad)')  # we already handled the x-label with ax1
ax2.scatter(times,angles[0,:], color=visualization.Lanc1,linestyle='dashed')
ax2.scatter(times,angles[1,:], color=visualization.Lanc2,linestyle='dashed')
ax2.scatter(times,angles[2,:], color=visualization.Lanc3,linestyle='dashed')
ax2.tick_params(axis='y')
ax2.set_ylim([-np.pi,np.pi])

ax1.set_xlabel('time (s)')
ax1.set_ylabel('Position (m)')
ax1.plot(positions[0,:], color=visualization.Lanc1)
ax1.plot(positions[1,:], color=visualization.Lanc2)
ax1.plot(positions[2,:], color=visualization.Lanc3)
ax1.set_ylim([-0.08,0.08])
ax1.tick_params(axis='y')


ax1.set_zorder(ax2.get_zorder()-1)
ax1.patch.set_visible(False)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

np.savetxt("Orientation" + str(orientation)+'.csv',remainingError[:,:,0],delimiter=',')
velocityProfile = SplineProfile.SplineProfile(env,simTime,env.times)

anim = visualization.animate2DPath(env,np.transpose(positions),transforms,controls,name="",savePlotSteps=velocityProfile.switchingTimes,save=True)
plt.figure()
plt.plot(np.linalg.norm(angles,axis=0))