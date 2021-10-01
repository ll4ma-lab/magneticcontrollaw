# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 10:55:02 2020

@author: tabor
"""


plt.rcParams["font.family"] = "Times New Roman"

fig,axs = plt.subplots(1,2)
colors = ['c','m','y','c','m','y']
stateNames = ['x','y','z','x','y','z']
plots = []
thickness = 1.0
linealpha = 1.0
dashedalpha = 0
#.set_title('position (m)')
axs[0].set_xlabel('Time (minutes)')
axs[0].set_ylabel('Position (m)')
ts = np.array(range(numSteps+1))/60.0
ts_desired = np.array(range(numSteps))/60.0 


newStates = np.copy(states[:6,:])
for i in range(numSteps):
    rot = visualization.getRotationAxis(states[3:6,i],1)
    for j in range(3):
        newStates[3+j,i] = np.arccos(rot[j][j,1])


visStates = newStates
#visStates = states



for i in range(3):
    plots.append(axs[0].plot(ts,np.transpose(visStates[i:i+1,:]),colors[i],linewidth=thickness,label = stateNames[i],alpha=linealpha)[0])
    #axs[0].plot(ts_desired,npdesiredStates[:,i:i+1,0],colors[i]+'--',linewidth=thickness,alpha=dashedalpha)
axs[0].legend(handles=plots, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
axs[1].set_xlabel('Time (minutes)')
axs[1].set_ylabel('Orientation (rad)')
for i in range(3,6):
    axs[1].plot(ts,np.transpose(visStates[i:i+1,:]),colors[i],label = stateNames[i],linewidth=thickness,alpha=linealpha)
    #axs[1].plot(ts_desired,npdesiredStates[:,i:i+1,0],colors[i]+'--',linewidth=thickness,alpha=dashedalpha)
fig.set_size_inches(7.5,2)
plt.tight_layout()
fig.savefig('orientation_off.pdf')