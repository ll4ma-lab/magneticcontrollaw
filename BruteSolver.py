# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:05:54 2019

@author: tabor
"""
try:
    print(imported)
except:
    print('derp')
    import numpy as np
    import itertools
    import tensorflow as tf
    import tensorflow_probability as tfp
    import warnings
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning) 
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
imported = True


class BruteSolver:
    def __init__(self,dims):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,use_per_session_threads=True))
        
        A_np = np.random.random((6,dims))
        B_np = np.random.random((6,dims)) * 0.1
        Y_np = np.random.random((6,1))
        
        self.A = tf.placeholder(tf.float32,shape = A_np.shape) 
        self.B = tf.placeholder(tf.float32,shape = B_np.shape)
        self.Y = tf.placeholder(tf.float32,shape = Y_np.shape)
        
        
        #build batched S matrix
        choices = [(-1,1)]*dims
        all_permutations = list(np.diag(x) for x in itertools.product(*choices))
        S_np = np.array(all_permutations[:100])
        
        S = tf.constant(S_np,tf.float32)
        B_ = tf.tile(tf.expand_dims(self.B,0),[S_np.shape[0],1,1])
        Y_ = tf.tile(tf.expand_dims(self.Y,0),[S_np.shape[0],1,1])
        
        paren = self.A+tf.matmul(B_,S)
        jacobians = tfp.math.pinv(paren)
     
        possible_Xs = tf.matmul(jacobians,Y_)
        SX = tf.matmul(S,possible_Xs)
        absX = tf.abs(possible_Xs)
        equals = tf.equal(SX,absX)
        reduced = tf.reduce_all(equals,[1])[:,0]
        self.true_Xs = tf.boolean_mask(possible_Xs,reduced)
    def solve(self,A_np,B_np,Y_np):
        output = self.sess.run(self.true_Xs,feed_dict={self.A:A_np,self.B:B_np,self.Y:Y_np})
        print('found ' + str(output.shape[0]) + ' solutions')
        if(output.shape[0]>0):
            return output[0]
        else:
            return None
