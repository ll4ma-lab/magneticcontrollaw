# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:43:16 2019

@author: tabor
"""
try:
    print(imported)
except:
    print('derp')
    import numpy as np
    import itertools
    import tensorflow as tf
    import time
    import tensorflow_probability as tfp
    import warnings
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning) 
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
    
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,use_per_session_threads=True))

from oct2py import octave

imported = True

def getAB(pose,magnets):
    sigma_copper = 5.8E+07;
    sigma_al = 3.77E+07;
    sigma = sigma_copper;
    do = 0.05;
    allFT,A,B = octave.calcEddyInput(magnets,sigma,do,nout=3)
    return A,B

magnets = 0.001 * np.array([[-130, 0, 0],[ 0, 0, -130],
                   [0, -130, 0],[ 0, 0, 130],
                   [130, 0, 0],[ 0, 130, 0]])
test_a,test_b = getAB(None,magnets)
# y = Ax + B|x|

#rewritten as 
# y = (A+BS)x where s is identity matrix with some negative signs
for n in [3,4,5,6]:
    print('devices ',n)
    dims = 3*n
    
    A_np = np.random.random((6,dims))
    B_np = np.random.random((6,dims)) * 0.1
    Y_np = np.random.random((6,1))
    
    A = tf.placeholder(tf.float32,shape = A_np.shape) 
    B = tf.placeholder(tf.float32,shape = B_np.shape)
    Y = tf.placeholder(tf.float32,shape = Y_np.shape)
    
    
    #build batched S matrix
    choices = [(-1,1)]*dims
    all_permutations = list(np.diag(x) for x in itertools.product(*choices))
    S_np = np.array(all_permutations)
    
    S = tf.constant(S_np,tf.float32)
    B_ = tf.tile(tf.expand_dims(B,0),[S_np.shape[0],1,1])
    Y_ = tf.tile(tf.expand_dims(Y,0),[S_np.shape[0],1,1])
    
    paren = A+tf.matmul(B_,S)
    jacobians = tfp.math.pinv(paren)
 
    possible_Xs = tf.matmul(jacobians,Y_)
    SX = tf.matmul(S,possible_Xs)
    absX = tf.abs(possible_Xs)
    equals = tf.equal(SX,absX)
    reduced = tf.reduce_all(equals,[1])[:,0]
    true_Xs = tf.boolean_mask(possible_Xs,reduced)
    start = time.clock()
    
    experiments = 100
    success = {}
    for i in range(20):
        success[i] = 0
    for i in range(experiments):
        A_np,B_np = getAB(None,magnets[:n])
        Y_np = np.random.random((6,1)) 
    
        #J = sess.run(jacobians,feed_dict={A:A_np,B:B_np,Y:Y_np})
        #poss = sess.run(possible_Xs,feed_dict={A:A_np,B:B_np,Y:Y_np})
        #bools = sess.run(equals,feed_dict={A:A_np,B:B_np,Y:Y_np})

        output = sess.run(true_Xs,feed_dict={A:A_np,B:B_np,Y:Y_np})
        
        success[output.shape[0]] = success[output.shape[0]] + 1
    print('in %d random experiments each had n solutions %s'%(experiments ,str(success)))
    totalTime = time.clock() - start
    Xs = output

    for i in range(Xs.shape[0]):
        #AX + B|x|
        YGivenX = np.add(np.matmul(A_np,Xs[i,:]),np.matmul(B_np,np.abs(Xs[i,:])))
        error = np.sum(np.subtract(Y_np,YGivenX))
        if error > 0.1:
            print('FAILED: error is ',error)
    
    
    individualTime = totalTime / experiments
    hz = 1.0/individualTime
    print('time per loop is ',individualTime)
    print('hertz of loop is ',hz)