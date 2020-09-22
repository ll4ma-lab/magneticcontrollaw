# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 18:27:19 2020

@author: tabor
"""

from visualization import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)
ax.set_aspect('equal')

addCube((0,0,0),0.1,1,ax)
drawSphere((0,0.5,0),0.1,ax)

vlen = 0.5


drawArrow((0,vlen,0),'g',ax)
drawArrow((0,-vlen,0),'b',ax)

lines = 60
for i in range(lines):
   angle = i* 2*np.pi/lines
   vec = (np.cos(angle),0,np.sin(angle))
   drawArrow(vec,'r',ax)