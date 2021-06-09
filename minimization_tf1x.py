# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:07:36 2021

@author: Griffin Tabor
"""
import tensorflow as tf
import numpy as np
import time

def forward(x):
    clippedTheta = tf.sigmoid(x) * 360 - 180
    
    return clippedTheta


def loss_fn(x):
    error = 0.7-tf.cos(x*np.pi/180.0)
    return 1e10*error*error

config = tf.ConfigProto(device_count = {'GPU': 0})

sess = tf.Session(config=config)

train_ops = []
costs = []
tf_vars = []
ys = []
for i in range(10):
    x = tf.Variable(0.001,dtype=tf.float64)
    y = forward(x)
    cost = loss_fn(y)
    
    ys.append(y)
    tf_vars.append(x)
    costs.append(cost)
    train_ops.append(tf.train.AdamOptimizer(5e-2).minimize(cost))

init_op = tf.global_variables_initializer()
sess.run(init_op)

init_op = [tf_vars[i].initializer for i in range(len(tf_vars))]
sess.run(init_op)

optimize_ops = [train_ops,costs,tf_vars,ys]

start = time.time()
for i in range(100):
    _,bestCosts,bestXs,bestYs = sess.run(optimize_ops)
    print(bestCosts)
print(time.time()-start)
print(bestYs)