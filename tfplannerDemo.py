# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:01:18 2020

@author: tabor
"""

try:
    print(imported)
except:
    print('derp')
    import numpy as np
    import tensorflow as tf
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,use_per_session_threads=True))
    
imported = True

import matplotlib.pyplot as plt

import numpy as np
from pythonModel import pythonModel
from exponentialSimulator import exponentialSimulator, sphereProperties
from environments import PlanarProblem
import visualization
import time
startRLTime = time.time()

env = PlanarProblem()
dt = 1

class tfSimulator(exponentialSimulator):
    def forwardPass(self,pose,control):
        newPose = tf.matmul(self.Ad, pose) + tf.matmul(self.Bd,control)
        return newPose
class basicModel():
    def __init__(self):
        npstuff = np.zeros((12,1))
        self.A = tf.Variable(npstuff,dtype=tf.float64)
        self.B = tf.Variable(npstuff,dtype=tf.float64)
        self.C = tf.Variable(npstuff,dtype=tf.float64)
        self.D = tf.Variable(npstuff,dtype=tf.float64)
    def forward(self,t):
        return self.A * (t*t*t) + self.B * (t*t) + self.C * t + self.D
    def getStart(self):
        return self.forward(0)
    def getEnd(self):
        self.T=1.0#tf.Variable(0,dtype=tf.float64)
        state = self.forward(self.T)
        return state
    def getBetween(self):
        ot = tf.Variable(0,dtype=tf.float64)
        t = tf.clip_by_value(ot,0,self.T)
        return self.forward(t),t
def error(state,target):    
    sqError = tf.squared_difference(state,target)
    return tf.math.reduce_sum(sqError,0)


simulator = tfSimulator(env.startState,dt,sphereProperties(8940,0.05)) #density, diamter
shape = basicModel()
start = env.startState
end = np.reshape(env.desiredState[:,0],(-1,1))

steps = 100
wrenchNP = np.zeros((6,1,steps))
wrenchesT = tf.Variable(wrenchNP,dtype=tf.float64)
AllStates = []
startShapeCost = error(start,shape.getStart()) 
endShapeCost = error(end,shape.getEnd())
cost = startShapeCost *100 + endShapeCost*100
percents = []
for i in range(steps):
    newState = simulator.simulate(wrenchesT[:,:,i])
    AllStates.append(newState)
    cState,cPercent = shape.getBetween()
    cost += error(newState,cState) + (1-cPercent) * 10.0**-3
    percents.append(cPercent)
reachGoalCost = error(newState,end)
cost +=  reachGoalCost * 10



train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
init_op = tf.global_variables_initializer()
sess.run(init_op)
wrenchT = []
stateT= []
costs = []
IndividualCosts = []
print('init time ' + str(time.time()-startRLTime))
optStart = time.time()
for i in range(500):
    if(i%100 is 0):
        print(str(i) +' ' + str(time.time()-optStart))
    _,outputCost,outputWrench,outputState,npPercents,npIndividualCosts = sess.run([train_op,cost,wrenchesT,AllStates,percents,[startShapeCost,endShapeCost,reachGoalCost]])
    costs.append(outputCost)
    stateT.append(outputState)
    wrenchT.append(outputWrench)
    IndividualCosts.append(npIndividualCosts)

wrenches = np.array(wrenchT)
states = np.array(stateT)
costs = np.array(costs)
Plots = {'Position (m)':np.transpose(states[-1,:,:2,0]),
         'Orientation (radians)':np.transpose(states[-1,:,5:6,0]),
         'Velocity (m/s)':np.transpose(states[-1,:,6:8,0]),
         "Force ":wrenches[-1,:3,0,:],
         "Torque ":wrenches[-1,3:,0,:]}
visualization.plotStuff(Plots,'demo')
anim = visualization.animateOptimization2DPath(env,states)
visualization.drawCost(costs)

    