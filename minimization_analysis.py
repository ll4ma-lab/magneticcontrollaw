# -*- coding: utf-8 -*- 
""" 
Created on Tue May 11 14:06:39 2021 
 
@author: Griffin Tabor 
""" 
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from problems.problem import Problem 
import numpy as np 

 
import torch 
import torch.multiprocessing  as multiprocessing

from solvers.line_search import GradientDescent, NewtonMethod, NewtonMethodWolfeLine 
import matplotlib.pyplot as plt 
import time 
from PDController import PDController
from torch.autograd import Variable

from environments import PlanarProblem,SpaceProblem
import SplineProfile
from discreteTimeSimulator import discreteTimeSimulator,sphereProperties


from models import multiMagModel,multiMagSolver

def optimize(decisionVariable): 
    solver = NewtonMethod(alpha, rho, prob,c1=c1,FP_PRECISION=FP_PRECISION) 
    solution, cost, converged, iterates = solver.optimize(decisionVariable, max_iterations=max_iterations)  
    return solution, cost, converged, iterates



env = SpaceProblem(4)
controller = PDController(env.startState,0.005,0.00001,0.005,0.00001,0.0,0.000)
velocityProfile = SplineProfile.SplineProfile(env,np.sum(env.times),env.times)


diameter = 0.04
objectProperties = sphereProperties(8940,diameter)



optimizerParams = {'alpha':1.0,'ro':0.9,'c1':1e-9,'FP_PRECISION': 1e-4}



newModel = multiMagModel(env.magnets*2,0.04)
control = [0,2,np.random.random()*40,np.random.random()*360-180]
control = [0, 2, 38.36714973571669, 172.5982599543529]
desired_wrench = newModel.forwardPass(np.zeros((3,1)), control)
#np.random.seed(473)

#desired_wrench = (np.random.random((6,1))-0.5)*1e-4    
#desired_wrench = (np.random.random((6,1))-0.5)*1e-7
#desired_wrench = (np.random.random((6,1))-0.5)*1e-7
#desired_wrench = (np.random.random((6,1))-0.5)*1e-7
#desired_wrench = (np.random.random((6,1))-0.5)*1e-7
results = []


BRUTEFORCECOLOR = False
if __name__ == '__main__':
    builder = multiMagSolver(newModel,'all')
    
    builder.modifyProblems(desired_wrench,np.zeros((3,1)))

    for i in range(len(builder.problems[:1])):
        prob = builder.problems[2]
        
    
    s = time.time()
    optimizer = NewtonMethod(problem=prob,**optimizerParams) 
    solution, cost, converged, iterates = optimizer.optimize(prob.initial_solution) 
    #print(solution,cost,converged,iterates.shape)

    strength = np.linspace(0,40,80)
    angles = np.linspace(-360,360,180)
    xx, yy = np.meshgrid(strength, angles)
    x=xx.reshape([-1])
    y=yy.reshape([-1])
    costs = []
    
    decisionVariablesList = []
    
    
    func = np.abs
    
    
    for i in range(x.shape[0]):
        decisionVariables = np.array([x[i],y[i]]).reshape((2,1))
        costs.append(func(prob.cost(decisionVariables)))
        decisionVariablesList.append(decisionVariables)
    
    
    
    ctx = multiprocessing.get_context('spawn')
    
    if BRUTEFORCECOLOR:
        with ctx.Pool(processes=8) as pool: 
           results.extend(pool.map(optimize, decisionVariablesList))
        
        colors = []
        badSolutions = []
        for result in results:
            solution, cost, converged, iterates = result
            if(cost<1e-5):
                colors.append('b')
            else:
                colors.append('y')
                badSolutions.append(solution)
    else:
        colors = ['b']*x.shape[0]
    
    bestarg = np.argmin(costs)
    
    print('bruteForce best ',x[bestarg],y[bestarg],' with cost ',costs[bestarg])
    print('newton method best ',solution[0],solution[1],'with cost ', cost,' ratio ', cost/costs[bestarg])
    
    foundCosts = []
    for j in range(iterates.shape[0]):
        foundCosts.append(func(prob.cost(iterates[j,:,:])))
    
    foundOldCosts = []
        
        
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x,y,costs,s=1,c=colors)
    ax.plot(np.abs(iterates[:,0,0]),iterates[:,1,0],foundCosts,c='r',markersize=200,linewidth=6)
    plt.show()

    print('elapsed time ',time.time()-s)