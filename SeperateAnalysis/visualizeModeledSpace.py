# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 18:27:19 2020

@author: tabor
"""

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from models import multiMagModel

from visualization import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_xlim(-0.5,0.5)
ax.set_ylim(-0.25,0.75)
ax.set_zlim(-0.5,0.5)
#ax.set_aspect('equal')

addCube((0,0,0),0.1,1,ax)
drawSphere((0,0.5,0),0.1,ax)

vlen = 0.5


drawArrow((0,vlen,0),'g',ax)
drawArrow((0,-vlen,0),'b',ax)

lines = 60
mags = [[0,-0.15,0]]
radius = 0.02
objectProperties = {'radius':radius}

for i in range(lines):
   angle = i* 2*np.pi/lines
   vec = (vlen*np.cos(angle),0,vlen*np.sin(angle))
   drawArrow(vec,'r',ax)
   
 
 
 
model = multiMagModel(mags,objectProperties,False)
scaleFactor = 1e3
position = np.zeros((3,1))
pointingForces = np.stack([model.forwardPass(position,[0,0,i,0]) for i in range(40+1)]).squeeze() * scaleFactor
ax.scatter(pointingForces[:,0],pointingForces[:,1]+0.5,pointingForces[:,2],c='g')

negPointingForces = np.stack([model.forwardPass(position,[0,1,i+0.5,0]) for i in range(40+1)]).squeeze() * scaleFactor
ax.scatter(negPointingForces[:,0],negPointingForces[:,1]+0.5,negPointingForces[:,2],c='b')



rhoForces = np.concatenate([np.stack([model.forwardPass(position,[0,2,i,j]) for i in range(1,40+1,5)]) for j in range(-180,180,6)],0).squeeze() * scaleFactor
ax.scatter(rhoForces[:,0],rhoForces[:,1]+0.5,rhoForces[:,2],c='purple')


arrow_color = 'k'
old_forces_color = Lanc1
new_forces_color = Lanc3

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_xlim(-0.5,0.5)
ax.set_ylim(-0.25,0.75)
ax.set_zlim(-0.5,0.5)
#ax.set_aspect('equal')

addCube((0,0,0),0.1,1,ax)

drawArrow((0,vlen,0),arrow_color,ax)
drawArrow((0,-vlen,0),arrow_color,ax)

lines = 60
mags = [[0,-0.15,0]]
radius = 0.02
objectProperties = {'radius':radius}

for i in range(lines):
   angle = i* 2*np.pi/lines
   vec = (vlen*np.cos(angle),0,vlen*np.sin(angle))
   drawArrow(vec,arrow_color,ax)


drawSphere((0,0.5,0),0.1,ax)
ax.scatter(pointingForces[:,0],pointingForces[:,1]+0.5,pointingForces[:,2],c=old_forces_color)
ax.scatter(negPointingForces[:,0],negPointingForces[:,1]+0.5,negPointingForces[:,2],c=old_forces_color)
ax.scatter(rhoForces[:,0],rhoForces[:,1]+0.5,rhoForces[:,2],c=old_forces_color)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])





sphereModel = multiMagModel(mags,objectProperties,True)

phiSet = [-90]#range(-180,180,2)
xiSet = range(-180,180,1)
powerSet = [0,40]
thetaSet = [-90,90]

sphereForces = np.concatenate([np.concatenate([np.stack([sphereModel.forwardPass(position,[0,i,j,k]) for i in [40]]) for j in phiSet],0) for k in xiSet],0).squeeze() * scaleFactor


rhoForces = np.concatenate([np.stack([model.forwardPass(position,[0,2,i,j]) for i in powerSet]) for j in thetaSet],0).squeeze() * scaleFactor
pointingForces = np.stack([model.forwardPass(position,[0,0,i,0]) for i in range(40+1)]).squeeze() * scaleFactor
print(np.arctan2(rhoForces[:, 2],rhoForces[:, 3])[-1]*2)



fig,ax = plt.subplots(figsize=(8,2))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xticks([])
ax.set_yticks([])

from matplotlib.patches import Polygon
from visualization import Lanc1,Lanc2,Lanc3
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], linewidth=5,c=old_forces_color,linestyle='dotted', label='Old Model'),
                   Patch(facecolor = new_forces_color,
                         label='New model')]

#ax.legend(handles=legend_elements, loc='best')


sortedSphere = sphereForces[sphereForces[:, 2].argsort(),1:3]
sortedrho = rhoForces[-rhoForces[:, 2].argsort(),1:3]
#negative so its + -> -  - ->+

allPoints = np.concatenate((sortedSphere, sortedrho))


plt.title('Slice')

bounds = (np.min(allPoints)*1.25,np.max(allPoints)*1.25)

#ax.set_xlim(bounds)
#ax.set_ylim(bounds)
ax.set_aspect('equal')
ax.plot(rhoForces[:2,2],rhoForces[:2,1], linewidth=5,c=old_forces_color,linestyle='dotted')
ax.plot(rhoForces[2:,2],rhoForces[2:,1], linewidth=5,c=old_forces_color,linestyle='dotted')
ax.plot(pointingForces[:,2],pointingForces[:,1], linewidth=5,c=old_forces_color,linestyle='dotted')

plt.tight_layout()
plt.savefig("old_modeled.png")

flippedAxis = np.stack([allPoints[:,1],allPoints[:,0]]).T
poly = Polygon(flippedAxis, facecolor = new_forces_color)
ax.add_patch(poly)

plt.savefig("new_and_old_modeled.png")


plt.show()