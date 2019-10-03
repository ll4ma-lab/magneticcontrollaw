# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:59:03 2019

@author: tabor
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def addCube(center,radius,ax):
    cube_definition = []
    center = np.array(center)
    first = tuple(center - radius)
    cube_definition.append(first)
    for i in range(3):
        offset= np.array([-radius,-radius,-radius])
        offset[i] = radius
        nextPoint = tuple(center+offset)
        cube_definition.append(nextPoint)
    if(center[2] == 0):
        color = (1,0,0,0.2)
    if(center[2] >0):
        color = (0,1,0,0.2)
    if(center[2] <0):
        color = (0,0,1,0.2)
    
    plot_cube(cube_definition,color,ax)
def drawCubes(centers,ax):
    for center in centers:
        addCube(center*10,0.15,ax)
    
def plot_cube(cube_definition,colorinfo,ax):
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]


    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    faces.set_facecolor(colorinfo)

    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0)

    ax.set_aspect('equal')
def drawArrows(vectors,errors,threshold,ax):
    ifSuccess = np.array(errors) < threshold
    for i in range(len(vectors)):
        vector = vectors[i]
        if(ifSuccess[i]):
            drawArrow(vector,'green',ax)
        else:
            drawArrow(vector,'red',ax)

def drawArrow(vector,color,ax):
    ax.quiver(0,0,0,vector[0],vector[1],vector[2],color=color,alpha=0.8,lw=3)
def visualizeEverything(magnets,vectors,errors,threshold,name):
      fig = plt.figure()
      fig.suptitle(name)
      ax = fig.add_subplot(111, projection='3d')  
      drawCubes(magnets,ax)
      drawArrows(vectors,errors,threshold,ax)
      plt.show()
