# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:43:16 2019

@author: tabor
"""
import numpy as np
import time
import BruteSolver
import gurobi
from oct2py import octave


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
for n in [1]:
    print('devices ',n)
    dims = 3*n
    solver = BruteSolver.BruteSolver(dims)
    
    secondSolver = gurobi.gurobiSolver(dims)
    start = time.clock()
    
    experiments = 10


    A,B = getAB(None,magnets[:n])
    for i in range(experiments):
        A = np.random.random((6,dims))
        B = np.random.random((6,dims)) * 0.1
        Y = np.random.random((6,1)) 
        output = solver.solve(A,B,Y)
        if output is not None:
            secondOutput = secondSolver.solve(A,B,Y)
            print(output)
            print(secondOutput)
            

        
    totalTime = time.clock() - start


    #test AX + B|x|
#    YGivenX = np.add(np.matmul(A_np,Xs[i,:]),np.matmul(B_np,np.abs(Xs[i,:])))
#    error = np.sum(np.subtract(Y_np,YGivenX))
#    if error > 0.1:
#        print('FAILED: error is ',error)
    
    
    individualTime = totalTime / experiments
    hz = 1.0/individualTime
    print('time per loop is ',individualTime)
    print('hertz of loop is ',hz)