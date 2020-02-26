# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:59:03 2019

@author: tabor
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def addCube(center,radius,op,ax):
    cube_definition = buildRectangularPrism(center,radius,radius,radius)
    if(center[2] == 0):
        color = (102/256.0,0,0,op)
    if(center[2] >0):
        color = (119/256.0,27/256.0,12/256.0,op)
    if(center[2] <0):
        color = (0,0,1,op)
    plot_cube(cube_definition,color,ax)
def addRect(center,l,w,h,color,ax):
    cube_definition = buildRectangularPrism(center,l,w,h)
    plot_cube(cube_definition,color,ax)

def buildRectangularPrism(center,l,w,h):
    cube_definition = []
    center = np.array(center)
    first = tuple(center - np.array([l,w,h]))
    cube_definition.append(first)
    for i in range(3):
        offset= np.array([-l,-w,-h])
        offset[i] = offset[i]*-1
        nextPoint = tuple(center+offset)
        cube_definition.append(nextPoint)
    return cube_definition
def drawCubes(allMagnets,magnets,ax):
    for magnet in allMagnets:
        op = 0.2
        if magnet in magnets:
            op = 0.8
        addCube(np.array(magnet)*1,0.05,op,ax)
    
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
    col = 'k'
    if(colorinfo[3]<0.5):
        col = 'w'
    faces = Poly3DCollection(edges, linewidths=1, edgecolors=col)
    faces.set_facecolor(colorinfo)

    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0)

    ax.set_aspect('equal')
def drawArrows(vectors,errors,threshold,ax,torque = False):
    ifSuccess = np.array(errors) < threshold
    sucess = np.sum(ifSuccess)
    print('succeeded on',sucess)
    print('failed on',len(ifSuccess)-sucess)
    for i in range(len(errors)):
        vector = vectors[i]
        if(ifSuccess[i]):
            drawArrow(vector,'green',ax,torque)
        else:
            drawArrow(vector,'red',ax,torque)
def visualize2DPath(environment,states,name='path'):
    allMagnets = environment.magnets
    magnets = environment.magnets
    fig = plt.figure()
    fig.suptitle(name)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(*environment.lims[1])
    ax.set_ylim(*environment.lims[1])
    ax.set_zlim(*environment.lims[1])
    
    drawCubes(allMagnets,magnets,ax)
    plt.plot(states[0,:],states[1,:],states[2,:]*0,linewidth=4,color='g')
    addRect((0,0,environment.lims[2][0]/2),l=environment.lims[0][1],w=environment.lims[1][1],
            h=environment.lims[2][1],color=(0,0,1,0.1),ax=ax)
    plt.show()
    
def plotStuff(dataDict,name):
    keys = dataDict.keys()
    fig,axs = plt.subplots(len(keys))
    fig.suptitle(name)
    for index in range(len(keys)):
        key = keys[index]
        axs[index].set_title(key)

        data = dataDict[key]
        for i in range(data.shape[0]):
            axs[index].plot(data[i,:])
def drawArrow(vector,color,ax,torque = False):
    if(torque):
        start_index = 3
    else:
        start_index = 0
    ax.quiver(0,0,0,vector[0+start_index],vector[1+start_index],vector[2+start_index],color=color,alpha=0.8,lw=3)
def visualizeEverything(allMagnets,magnets,vectors,errors,threshold,name,torque = False):


      fig = plt.figure()
      fig.suptitle(name)
      ax = fig.add_subplot(111, projection='3d')
      ax.set_xlim(-1.5,1.5)
      ax.set_ylim(-1.5,1.5)
      ax.set_zlim(-1.5,1.5)
      drawCubes(allMagnets,magnets,ax)
      drawArrows(vectors,errors,threshold,ax,torque)
      plt.show()
