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
ax.set_ylim(-0.5,1)
ax.set_zlim(-0.5,0.5)
#ax.set_aspect('equal')

addCube((0,0,0),0.1,1,ax)
drawSphere((0,0.5,0),0.1,ax)

vlen = 0.5


drawArrow((0,vlen,0),'g',ax)
drawArrow((0,-vlen,0),'b',ax)

lines = 60

for i in range(lines):
   angle = i* 2*np.pi/lines
   vec = (vlen*np.cos(angle),0,vlen*np.sin(angle))
   drawArrow(vec,'r',ax)
   

model = multiMagModel([[0,-0.5,0]],0.04,False)
scaleFactor = 5e7
position = np.zeros((3,1))
pointingForces = np.stack([model.forwardPass(position,[0,0,i,0]) for i in range(40)]).squeeze() * scaleFactor * 2
ax.scatter(pointingForces[:,0],pointingForces[:,1]+0.5,pointingForces[:,2],c='g')

negPointingForces = np.stack([model.forwardPass(position,[0,1,i+0.5,0]) for i in range(40)]).squeeze() * scaleFactor * 2
ax.scatter(negPointingForces[:,0],negPointingForces[:,1]+0.5,negPointingForces[:,2],c='b')

rhoForces = np.concatenate([np.stack([model.forwardPass(position,[0,2,i,j]) for i in range(40)]) for j in range(-180,180,3)],0).squeeze() * scaleFactor
ax.scatter(rhoForces[:,0],rhoForces[:,1]+0.5,rhoForces[:,2],c='r')



fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_xlim(-0.5,0.5)
ax.set_ylim(-0.5,1)
ax.set_zlim(-0.5,0.5)
#ax.set_aspect('equal')

addCube((0,0,0),0.1,1,ax)
drawSphere((0,0.5,0),0.1,ax)

model = multiMagModel([[0,-0.5,0]],0.04,True)


sphereForces = np.concatenate([np.concatenate([np.stack([model.forwardPass(position,[0,i,j,k]) for i in range(0,40,5)]) for j in range(-180,180,6)],0) for k in range(-180,180,6)],0).squeeze() * scaleFactor
ax.scatter(sphereForces[:,0],sphereForces[:,1]+0.5,sphereForces[:,2],c='r')
