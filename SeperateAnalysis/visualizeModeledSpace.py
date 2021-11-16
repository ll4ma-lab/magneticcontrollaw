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
import vispy
vispy.use("pyqt5")
from vispy import app, visuals, scene

# build your visuals, that's all
Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)

# The real-things : plot using scene
# build canvas
canvas = scene.SceneCanvas(keys='interactive', show=True)

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 8
view.camera.elevation = 90.0
view.camera.roll = 0.0
view.camera.azimuth = 0.0


# plot ! note the parent parameter






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
pointingForces = np.stack([model.forwardPass(position,[0,0,i,0]) for i in range(40+1)]).squeeze() * scaleFactor
ax.scatter(pointingForces[:,0],pointingForces[:,1]+0.5,pointingForces[:,2],c='g')

negPointingForces = np.stack([model.forwardPass(position,[0,1,i+0.5,0]) for i in range(40+1)]).squeeze() * scaleFactor
ax.scatter(negPointingForces[:,0],negPointingForces[:,1]+0.5,negPointingForces[:,2],c='b')



rhoForces = np.concatenate([np.stack([model.forwardPass(position,[0,2,i,j]) for i in range(40+1)]) for j in range(-180,180,3)],0).squeeze() * scaleFactor
ax.scatter(rhoForces[:,0],rhoForces[:,1]+0.5,rhoForces[:,2],c='r')



fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_xlim(-0.5,0.5)
ax.set_ylim(-0.5,1)
ax.set_zlim(-0.5,0.5)
#ax.set_aspect('equal')

addCube((0,0,0),0.1,1,ax)
drawSphere((0,0.5,0),0.1,ax)

sphereModel = multiMagModel([[0,-0.5,0]],0.04,True)

phiSet = [-90]#range(-180,180,2)
xiSet = range(-180,180,1)
powerSet = range(0,40+1)
thetaSet = [-90,90]

sphereForces = np.concatenate([np.concatenate([np.stack([sphereModel.forwardPass(position,[0,i,j,k]) for i in [40]]) for j in phiSet],0) for k in xiSet],0).squeeze() * scaleFactor


rhoForces = np.concatenate([np.stack([model.forwardPass(position,[0,2,i,j]) for i in powerSet]) for j in thetaSet],0).squeeze() * scaleFactor
pointingForces = np.stack([model.forwardPass(position,[0,0,i,0]) for i in range(40+1)]).squeeze() * scaleFactor

p1 = Scatter3D(parent=view.scene)
p1.set_gl_state('translucent', blend=True, depth_test=True)
p1.set_data(sphereForces[:,:3], symbol='o', size=10,
            edge_width=0.5)

p2 = Scatter3D(parent=view.scene)
p2.set_gl_state('translucent', blend=True, depth_test=True)
p2.set_data(rhoForces[rhoForces[:,0]>0,:3], symbol='o', size=20,     
            edge_width=0.5, edge_color='red',face_color='red')

            
p3 = Scatter3D(parent=view.scene)
p3.set_gl_state('translucent', blend=True, depth_test=True)
p3.set_data(pointingForces[:,:3], symbol='o', size=20,
            edge_width=0.5,face_color='green')


fig,ax = plt.subplots(figsize=(8,8))
#ax.plot(sphereForces[:,2],sphereForces[:,1])
#ax.plot(rhoForces[:,2],rhoForces[:,1])
from matplotlib.patches import Polygon
from visualization import Lanc1,Lanc2,Lanc3


sortedSphere = sphereForces[sphereForces[:, 2].argsort(),1:3]
sortedrho = rhoForces[-rhoForces[:, 2].argsort(),1:3]
#negative so its + -> -  - ->+

allPoints = np.concatenate((sortedSphere, sortedrho))

flippedAxis = np.stack([allPoints[:,1],allPoints[:,0]]).T
poly = Polygon(flippedAxis, facecolor = Lanc3)
ax.add_patch(poly)
ax.set_xlim([-3,3])
ax.set_ylim([-3,3])
ax.plot(rhoForces[:,2],rhoForces[:,1], linewidth=5,c=Lanc1,linestyle='dotted')
ax.plot(pointingForces[:,2],pointingForces[:,1], linewidth=5,c=Lanc1,linestyle='dotted')
plt.show()