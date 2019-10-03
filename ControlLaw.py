# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:43:16 2019

@author: tabor
"""
import numpy as np
import time
import BruteSolver
import gurobi
import visualization
from oct2py import octave


def getAB(pose,magnets):
    sigma_copper = 5.8E+07;
    sigma_al = 3.77E+07;
    sigma = sigma_copper;
    do = 0.05;
    allFT,A,B = octave.calcEddyInput(magnets,sigma,do,nout=3)
    return A,B
def evaluate(X,A,B,Y):
    if(X is None):
        temp = np.array(10.0**10)
        return temp
    YGivenX = np.add(np.matmul(A,X),np.matmul(B,np.abs(X)))
    errorVector = np.subtract(Y,YGivenX)
    error = np.dot(np.transpose(errorVector),errorVector)
    return error[0,0]
def timeMath(totalTime,experiments):
    individualTime = totalTime / experiments
    hz = 1.0/individualTime
    print('time per loop is ',individualTime)
    print('hertz of loop is ',hz)
    
All_magnets = 0.001 * np.array([[-130, 0, 0],[ 0, 0, -130],
                   [0, -130, 0],[ 0, 0, 130],
                   [130, 0, 0],[ 0, 130, 0]])
# y = Ax + B|x|

#rewritten as 
# y = (A+BS)x where s is identity matrix with some negative signs
random = False
threshold = 0.1

for n in [3,4,5,6]:
    print('devices ',n)
    dims = 3*n
    solver = BruteSolver.BruteSolver(dims)
    secondSolver = gurobi.gurobiSolver(dims)
    
    experiments = 12
    #unit directions
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
    magnets = All_magnets[:n]
    A,B = getAB(None,magnets)
    
    
    results = []
    start = time.clock()
    for i in range(experiments):
        Y = Ys[i]
        secondOutput = secondSolver.solve(A,B,Y,i+1)
        error = evaluate(secondOutput,A,B,Y)
        print("gurobi " ,error)
        results.append(error)
    totalTime = time.clock() - start
    timeMath(totalTime,experiments)
    visualization.visualizeEverything(magnets,Ys,results,threshold,'gurobi')
    
    
    results = []
    print('gpu')
    for i in range(experiments):
        Y = Ys[i]
        output = solver.solve(A,B,Y)
        error = evaluate(output,A,B,Y)
        results.append(error)
        print('GPU ', error )
    totalTime = time.clock() - start
    timeMath(totalTime,experiments)
    visualization.visualizeEverything(magnets,Ys,results,threshold,'GPU')


    
