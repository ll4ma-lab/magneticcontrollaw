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

env = PlanarProblem()
dt = 1

class tfSimulator(exponentialSimulator):
    def forwardPass(self,pose,control):
        newPose = tf.matmul(self.Ad, pose) + tf.matmul(self.Bd,control)
        return newPose



simulator = tfSimulator(env.startState,dt,sphereProperties(8940,0.05)) #density, diamter
normalSimulator = exponentialSimulator(env.startState,dt,sphereProperties(8940,0.05)) #density, diamter
start = env.startState
end = np.reshape(env.desiredState[:,0],(-1,1))

steps = 1000
wrenchNP = np.zeros((6,1,steps))
wrenchesT = tf.Variable(wrenchNP,dtype=tf.float64)
AllStates = []
for i in range(steps):
    newState = simulator.simulate(wrenchesT[:,:,i])
    AllStates.append(newState)
sqError = tf.squared_difference(newState,end)
cost = tf.math.reduce_sum(sqError,0)







train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
init_op = tf.global_variables_initializer()
sess.run(init_op)
wrenchT = []
stateT= []
for i in range(500):
    if(i%100 is 0):
        print(i)
    _,outputWrench,outputState = sess.run([train_op,wrenchesT,AllStates])
    stateT.append(outputState)
    wrenchT.append(outputWrench)

wrenches = np.array(wrenchT)
states = np.array(stateT)

Plots = {'Position (m)':np.transpose(states[-1,:,:2,0]),
         'Orientation (radians)':np.transpose(states[-1,:,5:6,0]),
         'Velocity (m/s)':np.transpose(states[-1,:,6:8,0]),
         "Force ":wrenches[-1,:3,0,:],
         "Torque ":wrenches[-1,3:,0,:]}
visualization.plotStuff(Plots,'demo')
anim = visualization.animateOptimization2DPath(env,states)

    