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
def evaluate(X,A,B,Y):
    YGivenX = np.add(np.matmul(A,X),np.matmul(B,np.abs(X)))
    errorVector = np.subtract(Y,YGivenX)
    error = np.dot(np.transpose(errorVector),errorVector)
    return error[0,0]
magnets = 0.001 * np.array([[-130, 0, 0],[ 0, 0, -130],
                   [0, -130, 0],[ 0, 0, 130],
                   [130, 0, 0],[ 0, 130, 0]])
test_a,test_b = getAB(None,magnets)
# y = Ax + B|x|

#rewritten as 
# y = (A+BS)x where s is identity matrix with some negative signs
random = False


for n in [3,4,5]:
    print('devices ',n)
    dims = 3*n
    solver = BruteSolver.BruteSolver(dims)
    
    secondSolver = gurobi.gurobiSolver(dims)
    
    experiments = 12
    if(not random):
        Ys = []
        for i in range(6):
            currY = np.zeros((6,1))
            currY[i] = 1
            Ys.append(np.copy(currY))
            currY[i] = -1
            Ys.append(np.copy(currY))
    else: 
        Ys = (np.random.random((experiments,6,1)) -0.5)*2

    A,B = getAB(None,magnets[:n])
    start = time.clock()
    for i in range(experiments):
        Y = Ys[i]
        output = solver.solve(A,B,Y)
        if output is not None:
            print('GPU ',evaluate(output,A,B,Y))
        secondOutput = secondSolver.solve(A,B,Y,i+1)
        print("gurobi " ,evaluate(secondOutput,A,B,Y))
        
    totalTime = time.clock() - start
    
    
    individualTime = totalTime / experiments
    hz = 1.0/individualTime
    print('time per loop is ',individualTime)
    print('hertz of loop is ',hz)