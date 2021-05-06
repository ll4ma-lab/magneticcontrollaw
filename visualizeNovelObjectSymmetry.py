# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:56:38 2021

@author: Griffin Tabor
"""

from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from cylinderModel import getOmniFrame
from visualization import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from scipy.spatial.transform import Rotation as R
import se3Utils


def simpleTFBuild(euler,translation):
    worldToBodyRotation = R.from_euler('xyz',euler)
    worldToBodyR = worldToBodyRotation.as_dcm()

    bodyToWorldR = np.transpose(worldToBodyR)
    bodyToWorld = np.eye(4)
    bodyToWorld[:3,:3] = bodyToWorldR
    bodyToWorld[:3,3] = translation
    return bodyToWorld
def sphereicalToCartesian(theta,phi,r):
    x = r*np.cos(theta)*np.sin(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(phi)
    position = np.array([x,y,z,1]).reshape((4,1))
    return position
def cartesianToSpherical(position):
    p = position.flatten()
    r = np.sqrt(np.sum(p**2))
    phi = np.arccos(p[2]/r)
    theta = np.arctan2(p[1],p[0])
    return theta,phi,r
def buildControlVector(theta,phi):
    vector = sphereicalToCartesian(theta,phi,1)
    axisShift = np.transpose(np.array([[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1]]))
    newVector = np.dot(axisShift,vector)
    return newVector
    
def rotX(theta):
    return np.array([[1,0,0,0],
              [0,np.cos(theta),-np.sin(theta),0],
              [0,np.sin(theta),np.cos(theta),0],
              [0,0,0,1]])
def rotY(theta):
    return np.array([[np.cos(theta),0,np.sin(theta),0],
              [0,1,0,0],
              [-np.sin(theta),0,np.cos(theta),0],
              [0,0,0,1]])
def rotZ(theta):
    return np.array([[np.cos(theta),-np.sin(theta),0,0],
              [np.sin(theta),np.cos(theta),0,0],
              [0,0,1,0],
              [0,0,0,1]])
def buildXAlignRot(transform,omniTF):
    objectToModelFrame = np.dot(np.linalg.inv(transform),omniTF)
    o_theta,o_phi,_ = cartesianToSpherical(objectToModelFrame[:3,0])
    axisShift = np.array([[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1]])
    offsetRot = np.dot(axisShift,np.dot(rotX(o_theta),rotZ(o_phi)))
    return offsetRot,o_theta,o_phi


global transform, controlArrow

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)
addCube((0,0,0),0.1,0.3,ax)
your_mesh = mesh.Mesh.from_file("../../Downloads/USB-extension-cord_clip/files/usb-clamp.stl")
your_mesh.points *=0.003


transform = simpleTFBuild(np.array([0.8,0.3,0.5]),np.array([0.5,0.5,0.5]))
your_mesh.transform(transform)

objectAxisLines = drawOrigin(ax,0.3)


omniAxisLines = drawOrigin(ax,0.3)

omniR = getOmniFrame(-transform[:3,3])
omniTF = np.eye(4)
omniTF[:3,:3] = omniR

updateAxesWithTF(omniTF,omniAxisLines,0.3)
updateAxesWithTF(transform,objectAxisLines,0.3)

positionLine = ax.plot(*[[0,transform[i,3]] for i in range(3)],c='black')[0]

mesh_collection = mplot3d.art3d.Poly3DCollection(your_mesh.vectors)
ax.add_collection3d(mesh_collection)



theta = np.deg2rad(45)
phi = np.deg2rad(60)
r = 0.5


controlVec = np.dot(omniTF,buildControlVector(theta, phi))
controlArrow = drawArrow(controlVec,'r',ax)

# Auto scale to the mesh size
scale = your_mesh.points.flatten()

# Show the plot to the screen
plt.show()

test = drawOrigin(ax,0.3)

offsetRot,o_theta,o_phi = buildXAlignRot(transform,omniTF)
updateAxesWithTF(np.dot(transform,offsetRot),test,0.5)

print(transform)

def update_anim(num,mesh,mesh_collection,omniAxisLines,objectAxisLines,positionLine):
    global transform,controlArrow,modeSwitchTransform
    if( num <100):
        newRot = rotZ(0.01)
        mesh.transform(newRot)
        transform = np.dot(newRot,transform)
    else:
        #newRot = rotX(0.01)
        rotAxis = getOmniFrame(-transform[:3,3])[:3,0] #get its X axis
        newRot = se3Utils.matrixExponential(se3Utils.buildSkew(rotAxis),0.01)
        goalTransform = np.copy(transform) #spin in place
        goalTransform[:3,:3] = np.dot(newRot[:3,:3],transform[:3,:3]) #reset modified position
        
        deltaTransform = np.dot(goalTransform,np.linalg.inv(transform))
        mesh.transform(deltaTransform)
        transform = goalTransform
        

    mesh_collection.set_verts(mesh.vectors)
    
    omniR = getOmniFrame(-transform[:3,3])
    omniTF = np.eye(4)
    omniTF[:3,:3] = omniR
    
    
    updateAxesWithTF(omniTF,omniAxisLines,0.3)
    updateAxesWithTF(transform,objectAxisLines,0.3)

    offsetRot,o_alpha,o_beta = buildXAlignRot(transform,omniTF)
    #print(num,o_alpha,o_beta)

    objectModelAlignedFrame = np.dot(transform,offsetRot)
    updateAxesWithTF(objectModelAlignedFrame,test,0.0)


    modelToModelAligned = np.dot(objectModelAlignedFrame,np.linalg.inv(omniTF))
    skew,theta_object = se3Utils.matrixLog(modelToModelAligned[:3,:3])
    finalDofAlignmentAxis = se3Utils.skewToVector(skew)[:,0]
    sameAxis = np.dot(omniTF[:3,0],finalDofAlignmentAxis) # are the thetas pointing in the same axis or anti parallel
    #print('pointing axis ' + str(omniTF[:3,0]) + 'finalRotAxis  ' + str(finalDofAlignmentAxis) + ' amount ' + str(theta_object))
    theta_control = theta + sameAxis * theta_object 
    #print(sameAxis,theta_control,phi)
    
    positionLine.set_data(*[[0,transform[i,3]] for i in range(2)])
    positionLine.set_3d_properties([0,transform[2,3]])
    
    
    controlVec = np.dot(omniTF,buildControlVector(theta_control, phi))
    controlArrow.remove()
    controlArrow = drawArrow(controlVec,'r',ax)

for i in range(360):
    controlVec = np.dot(omniTF,buildControlVector(np.deg2rad(i),np.pi))
    #controlArrow.remove()
    #drawArrow(controlVec,'r',ax)      

stateSpaceAnimation = animation.FuncAnimation(fig, update_anim, 200, fargs=(your_mesh,mesh_collection,omniAxisLines,objectAxisLines,positionLine),
                                       interval=1, blit=False)

print(transform)
