# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:59:03 2019

@author: tabor
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation




Lanc1 = (0,0.4470,0.7410)
Lanc2 = (0.8500,0.3250,0.0980)
Lanc3 = (0.9290,0.6940,0.1250)

def addCube(center,radius,op,ax):
    cube_definition = buildRectangularPrism(center,radius,radius,radius)
    if(center[2] == 0):
        color = (102/256.0,0,0,op)
    if(True or center[2] >0):
        color = (119/256.0,27/256.0,12/256.0,op)
    elif(center[2] <0):
        color = (0,0,1,op)
    faces = plot_cube(cube_definition,color,ax)
    return faces
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
def drawCubes(allMagnets,magnets,ax,radius=0.065):
    allFaces = []
    for magnet in allMagnets:
        op = 0.2
        if magnet in magnets:
            op = 0.8
        face = addCube(np.array(magnet)*1,radius,op,ax)
        allFaces.append(face)
    return allFaces
    
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
    return faces
    #ax.set_aspect('equal')
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
def drawSphere(center,radius,ax):
    # draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v) * radius + center[0]
    y = np.sin(u)*np.sin(v) * radius + center[1]
    z = np.cos(v) * radius + center[2]
    ax.plot_wireframe(x, y, z, color="orange")
    
def visualize2DPath(environment,states,name='path'):
    allMagnets = environment.magnets
    magnets = environment.magnets
    fig = plt.figure()
    fig.suptitle(name)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(*environment.visualizationBoundaryY)
    ax.set_ylim(*environment.visualizationBoundaryY)
    ax.set_zlim(*environment.visualizationBoundaryY)
    #ax.set_aspect('equal')

    drawCubes(allMagnets,[],ax,environment.radius)
    plt.plot(states[0,:],states[1,:],states[2,:],linewidth=4,color='g')
    addRect((0,0,environment.lims[2][0]/2),w=environment.lims[0][1],l=environment.lims[1][1],
            h=environment.lims[2][1],color=(0,0,1,0.1),ax=ax)
            
    ax.view_init(elev=environment.el, azim=environment.az)

    plt.show()

def animate2DPath(environment,positions,transforms,controls,name='path',save=False):
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_xlim(*environment.visualizationBoundaryY)
    ax.set_ylim(*environment.visualizationBoundaryY)
    ax.set_zlim(*environment.visualizationBoundaryY)
    ax.set_aspect('equal')

    cubes = drawCubes(environment.magnets,[],ax,environment.radius)
    addRect((0,0,environment.lims[2][0]/2),w=environment.lims[0][1],l=environment.lims[1][1],
            h=environment.lims[2][1],color=(0,0,1,0.1),ax=ax)
    ax.set_title(name)


    def update_lines(num, positions, transforms, controls, traj_plot, scatter,lines,cubes):
        #num = num + states.shape[1]/2
        for i in   range(len(cubes)):
            cube = cubes[i]
            colorinfo = [119/256.0,27/256.0,12/256.0,0.2]
            #print(controls[num][0],i)
            if(controls[num][0] ==i):
                colorinfo[3] = 0.8
            cube.set_facecolor(colorinfo)

        #traj_plot.set_data(np.transpose(positions[:num+1,0:2]))
        #traj_plot.set_3d_properties(positions[:num+1,2])
        scatter._offsets3d = (positions[num,0:1],positions[num,1:2],positions[num,2:3])
        for i in range(3):
            vector = np.zeros((4,2))
            vector[i,0] = 0.1
            vector[3,:] = 1
            vertices = np.dot(transforms[num],vector)
            lines[i].set_data(vertices[0:2, :])
            lines[i].set_3d_properties(vertices[2, :])
        return traj_plot,scatter,lines
    traj_plot = ax.plot(positions[0:1,0], positions[0:1,1], positions[0:1,2],linewidth=2,color='g')[0] 
    
    lines = [ax.plot(positions[0:1,0], positions[0:1,1], positions[0:1,2],linewidth=2,color=Lanc1)[0] ,
            ax.plot(positions[0:1,0], positions[0:1,1], positions[0:1,2],linewidth=2,color=Lanc2)[0] ,
            ax.plot(positions[0:1,0], positions[0:1,1], positions[0:1,2],linewidth=2,color=Lanc3)[0] ]
    
    Fixedlines = [ax.plot((0,0.1), (0,0), (0,0),linewidth=2,color=Lanc1)[0] ,
            ax.plot((0,0), (0,0.1), (0,0),linewidth=2,color=Lanc2)[0] ,
            ax.plot((0,0), (0,0), (0,0.1),linewidth=2,color=Lanc3)[0] ]
    scatter = ax.scatter(positions[0:1,0], positions[0:1,1], positions[0:1,2],linewidth=10,color='orange') 
    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, positions.shape[0], fargs=(positions, transforms,controls, traj_plot, scatter,lines,cubes),
                                       interval=1, blit=False)
    
    ax.view_init(elev=environment.el, azim=environment.az)
    plt.show()
    print('newer dpi')
    if(save):
        line_ani.save('animation.mp4', writer='ffmpeg', fps=20,dpi=400)
    return line_ani
def animateOptimization2DPath(environment,states,name='path'):

    fig = plt.figure()
    
    ax = fig.gca(projection="3d")
    
    ax.set_xlim(*environment.visualizationBoundary)
    ax.set_ylim(*environment.visualizationBoundary)
    ax.set_zlim(*environment.visualizationBoundary)
    
    drawCubes(environment.magnets,environment.magnets,ax)
    addRect((0,0,environment.lims[2][0]/2),l=environment.lims[0][1],w=environment.lims[1][1],
            h=environment.lims[2][1],color=(0,0,1,0.1),ax=ax)
    ax.set_title(name)
    
    line, = ax.plot([], [], [], lw=2,color='green')
    def animate(i):
        x = states[i,:,0]
        y = states[i,:,1]
        z = states[i,:,2]
        line.set_data(x, y)
        line.set_3d_properties(z)
        fig.canvas.draw()
        return line
    anim = animation.FuncAnimation(fig, animate, frames=states.shape[0], interval=2)
    
    plt.show()
    return anim
def drawCost(costs):
    newFig = plt.figure()
    plt.plot(range(costs.shape[0]),costs)
    plt.grid()
    plt.yscale("log")
    newFig.axes[0].set_title("loss")
    plt.show()
def plotStuff(dataDict,name):
    keys = dataDict.keys()
    fig,axs = plt.subplots(len(keys))
    fig.suptitle(name)
    for index in range(len(keys)):
        key = list(keys)[index]
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
