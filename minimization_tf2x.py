# -*- coding: utf-8 -*-
"""
Created on Fri May  7 10:57:50 2021

@author: Griffin Tabor
"""

import tensorflow as tf
import numpy as np
import time
@tf.function
def forward(x):
    clippedTheta = tf.sigmoid(x) * 720 - 360
    
    return clippedTheta

@tf.function
def loss_fn(x):
    error = 0.7-tf.cos(x*np.pi/180.0)
    return 1e10 * error*error

def trainLoop(opt,x):
    for j in range(100):
        with tf.GradientTape() as tape:
            y = forward(x)
            loss = loss_fn(y)
        input_vars = [x]
        gradients = tape.gradient(loss,input_vars)
        zipped = zip(gradients,input_vars)
        opt.apply_gradients(zipped)
    return x

def train():
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    x = tf.Variable(0.001,dtype=tf.float64)
    return trainLoop(opt,x)


train()
start = time.time()
with tf.device('/job:localhost/replica:0/task:0/device:CPU:0'): 
    for i in range(10):
        x= train()
    print(time.time()-start)
    print(forward(x))