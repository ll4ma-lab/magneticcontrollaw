# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:54:42 2020

@author: tabor
"""
from cylinderModel import cylinderSolver,cylinderModel

from oct2py import octave
import numpy as np
import matplotlib.pyplot as plt
from environments import PlanarProblem
import time 
def evaluateTheta(theta):
    thetarad = np.deg2rad(theta)
    Rx=np.array([[1,0,0],
              [0,np.cos(thetarad) , -np.sin(thetarad)],
              [0,np.sin(thetarad) , np.cos(thetarad)]])
    R = np.dot(model.getOmniFrame(pos),Rx)
    
    rotatedVector = np.concatenate([np.dot(R,cart[:3,0]),np.dot(R,cart[3:,0])])
    return rotatedVector

env = PlanarProblem()
model = cylinderModel(env.magnets)
solver = cylinderSolver(env.magnets)



diamter = 0.05
freq = 10.0
power = 40.0
state = env.startState

state[1] = -0.16
state[2] = 0.12
position = state[0:3]
pos = (np.array(env.magnets) - np.transpose(position)).tolist()[0]

cycle,cart = model.modelCall(pos,power)

desired = np.zeros((6,1))
desired[1] = 0.01


costs=[]
thetas = range(-180,180)
for theta in thetas:
    
    rotatedVector = evaluateTheta(theta)
    cost = np.linalg.norm(desired[:,0]-rotatedVector)**2
    costs.append(cost)
print(thetas[np.argmin(costs)])
print(np.min(costs))
plt.plot(thetas,np.array(costs) - np.min(costs))

plt.yscale("log")
plt.show()
s = time.time()
a,b,c = solver.solveWrench(model,state,desired)
print(time.time()-s)
