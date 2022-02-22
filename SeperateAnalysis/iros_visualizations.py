#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 12:51:19 2022

@author: tabor
"""

import sys
sys.path.append("..") # Adds higher directory to python modules path.


from visualization import drawCubes, addRect, drawSphere
from environments import SpaceProblem
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

environment = SpaceProblem(4)
environment.lims = [[np.min(environment.magnets),np.max(environment.magnets)]] * 3
environment.visualizationBoundaryY = (-0.2,0.2)

#environment.magnets = (np.array(((np.sqrt(8/9),0,1/3),(-np.sqrt(2/9),np.sqrt(2/3),1/3),(-np.sqrt(2/9),-np.sqrt(2/3),1/3),(0,0,-1))) * np.max(environment.magnets)).tolist() 

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_xlim(*environment.visualizationBoundaryY)
ax.set_ylim(*environment.visualizationBoundaryY)
ax.set_zlim(*environment.visualizationBoundaryY)
cubes = drawCubes(environment.magnets,environment.magnets,ax,environment.radius/3)

addRect((0,0,0),w=environment.lims[0][1],l=environment.lims[1][1],
        h=environment.lims[2][1],color=(0,0,1,0.1),ax=ax)
ax.view_init(elev=environment.el, azim=environment.az)


fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_xlim(*environment.visualizationBoundaryY)
ax.set_ylim(*environment.visualizationBoundaryY)
ax.set_zlim(*environment.visualizationBoundaryY)

pts = np.array(environment.magnets)
from scipy.spatial import ConvexHull

hull = ConvexHull(pts)

edges = [pts[simplices].tolist() for simplices in hull.simplices ]

col = 'k'
colorinfo = [0.4]*4
faces = Poly3DCollection(edges, linewidths=1, edgecolors=col)
faces.set_facecolor(colorinfo)
ax.add_collection3d(faces)
# Make axis label
ax.view_init(elev=-169, azim=-42)
cubes = drawCubes(environment.magnets,environment.magnets,ax,environment.radius*0.4)

#convex_hull_point = np.mean(pts[hull.simplices[0]],axis=1)
#drawSphere(convex_hull_point*2,0.02,ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

