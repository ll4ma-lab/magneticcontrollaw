# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:41:12 2019

@author: tabor
"""
import numpy as np
import time
from oct2py import octave
import numpy as np
import tensorflow as tf
class cylinderSolver():
    def __init__(self,magnets):
        self.magnets = magnets
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,use_per_session_threads=True))
    
    
    def solveWrench(self,model,state,Y):
        s = time.time()
        
        power = tf.Variable(5,dtype=tf.float64)
        clippedPower = tf.clip_by_value(power,0,400)
        
        theta = tf.Variable(0,dtype=tf.float64)
        clippedTheta = theta#tf.clip_by_value(theta,-180,180)
        
        radians = clippedTheta*np.pi/180.0        
        
        position = state[0:3]
        chosenMagnetPose = (np.array(self.magnets) - np.transpose(position)).tolist()[0]
        omniFrame = model.getOmniFrame(chosenMagnetPose)
        
        print(1,time.time()-s)
        s = time.time()

        
        _,cart = model.modelCall(chosenMagnetPose,1)
        print(2,time.time()-s)
        s = time.time()

        
        one = tf.constant(1.0,dtype=tf.float64)
        zero = tf.constant(0.0,dtype=tf.float64)
        Rx=np.array([[one,zero,zero],
                  [zero,tf.cos(radians) , -tf.sin(radians)],
                  [zero,tf.sin(radians) , tf.cos(radians)]])
        Rx_t = tf.stack([tf.stack(Rx[0].tolist()),tf.stack(Rx[1].tolist()),tf.stack(Rx[2].tolist())])
        omniFrame_t = tf.convert_to_tensor(omniFrame)
        R = tf.matmul(omniFrame_t,Rx_t)
        
        cart = cart*clippedPower * clippedPower
        FT = tf.concat([tf.matmul(R,cart[:3,:1]),tf.matmul(R,cart[3:,:1])],0)
        
        errors = FT-Y
#
#        cost = errors[0] * errors[0] 
        cost = tf.matmul(tf.transpose(errors),errors)
        print(3,time.time()-s)
        s = time.time()


        train_op = tf.train.AdamOptimizer(1e1).minimize(cost)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        
        print(4,time.time()-s)
        s = time.time()

        for i in range(100):
            _,bestCost,closestFT,bestTheta, p = self.sess.run([train_op,cost,FT,theta,power])
            #print(bestCost[0,0],closestFT[:,0].tolist(),bestTheta,p)
        print(5,time.time()-s)
        return bestCost,closestFT,bestTheta

        
    
    
class cylinderModel(object):
    def __init__(self,magnets):
        self.magnets = magnets
        self.dims = len(magnets)*3
        self.time = 0.0
        sigma_copper = 5.8E+07;
        sigma_al = 3.77E+07;
        self.sigma = sigma_copper;
        self.do = 0.05;
        self.freq = 15
        
    def modelCall(self,chosenMagnetPose,power):
        cycle,cart = octave.eddyFTRotated(np.linalg.norm(chosenMagnetPose),power,self.freq,self.do/2.0,nout=2)
        return cycle,cart
    def forwardPass(self,pose,control):
            #control[0] = magnetNumber
            #control[1] = modelNumber
            #control[2] = power
            #control[3] = theta
        position = pose[0:3]
        chosenMagnetPose = (np.array(self.magnets) - np.transpose(position)).tolist()[control[0]]
        omniFrame = self.getOmniFrame(chosenMagnetPose)
        _,cart = self.modelCall(chosenMagnetPose,power)
        if(control[1]==0):
            theta = control[3]
            thetarad = np.deg2rad(theta)
            Rx=np.array([[1,0,0],
                      [0,np.cos(thetarad) , -np.sin(thetarad)],
                      [0,np.sin(thetarad) , np.cos(thetarad)]])
            R = np.dot(omniFrame,Rx)
            
            FT = np.concatenate([np.dot(R,cart[:3,1]),np.dot(R,cart[3:,1])])
        if(control[1]==1):
            FT = np.concatenate([np.dot(omniFrame,cart[:3,1]),np.dot(omniFrame,cart[3:,1])])
        if(control[1]==2):
            FT = np.concatenate([np.dot(omniFrame,cart[:3,2]),np.dot(omniFrame,cart[3:,2])])

        return FT
    def getOmniFrame(self,pos):
        dist = np.linalg.norm(pos)
        
        omni_X = -1*(pos/dist);
        
        # Set default Z vector as upward
        Z = np.array([0, 0, 1])
        #
        # Determine if default Z and X are the same
        if np.all(abs(omni_X) == abs(Z)):
            #         fprintf('X and Z are the same vectors, new perpendicular Z chosen [0 1 0]\n');
            #print('X and Z are the same vectors, new perpendicular Z chosen [0 1 0]')
            Z = np.array([0, 1, 0])#set Z to be something else if already equal
        
        # Determine Z component perpendicular to Y axis
        if np.dot(omni_X,Z) == 0:
            #         fprintf('Z and X are already perpendicular\n');
            #print('Z and X are already perpendicular')
            omni_Z = Z;
        else:
            omni_Z = np.cross(omni_X,Z)
            omni_Z = omni_Z/np.sqrt(np.dot(omni_Z,omni_Z))
                
        
        # Obtain Y axis by crossing Z and X
        omni_Y = np.cross(omni_Z,omni_X);
        # Combine to build rotation matrix
        omniFrame = np.transpose(np.array([omni_X, omni_Y, omni_Z]));
        return omniFrame
