#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 14:02:43 2021

@author: tabor
"""

#online optimization

from ll4ma_opt.solvers.line_search import NewtonMethod, GradientDescent 
 
from ll4ma_opt.problems import Problem  
import torch 
import numpy as np 
import matplotlib.pyplot as plt 

np.random.seed(473)
class simpleOnlineProblem(Problem):  
    def __init__(self):  
        super().__init__()     
        self.initial_solution = np.ones((9,1)) * 0.05 
    def updateVals(self,y,x): 
        self.y = torch.tensor(y) 
        self.x = torch.tensor(x) 
    def tensor_cost(self, params): 
        predicted = params.reshape((3,3))@self.x 
         
        error = (self.y - predicted)[:,0] 
        loss = torch.dot(error,error) 
        return loss 
class regularizedProblem(Problem):
    def __init__(self,prob,lamb):
        super().__init__()
        self.updateVals = prob.updateVals
        self.normalProb = prob
        self.initial_solution = prob.initial_solution
        self.lamb = torch.tensor(lamb)
    def tensor_cost(self,params):
        return self.normalProb.tensor_cost(params) + self.lamb * params.T @ params
class onlineMomentum:
    def __init__(self,problem,step,smoothness):
        self.x = np.copy(problem.initial_solution)
        self.current_step = np.zeros(self.x.shape)
        self.prob = problem
        self.stepSize = step
        self.smoothness = smoothness
        
        self.iterates = []
        self.costs = [] 
        
    def step(self,updateVals):
        self.prob.updateVals(*updateVals)
        self.current_step = self.smoothness*self.current_step + (1-self.smoothness) * (-prob.cost_gradient(self.x))
        self.x += self.stepSize * self.current_step
        self.iterates.append(np.copy(self.x))
        
        self.costs.append(self.prob.cost(self.x))
        return self.x,self.iterates

trueParam = (np.random.random((3,3)) - 0.5) 
def getSample(): 
    x = np.random.random((3,1)) 
    y = trueParam @ x + (np.random.normal(0,0.01,(3,1))) 
    return y,x 
 
 
step = 1e-2
 
optimizerParams = {'alpha':step,'ro':0.9,'min_alpha': step/2} 
prob = regularizedProblem(simpleOnlineProblem(),1e-2)
optimizer = GradientDescent(problem=prob,**optimizerParams) 
onlineOptimizer = onlineMomentum(prob,step,0.9)
 
solution = prob.initial_solution 
iterates = [] 
costs = []
for j in range(1000): 
    sample = getSample()
    prob.updateVals(*sample) 
    solution, cost, converged, iterate = optimizer.optimize(solution,max_iterations=1) 
    costs.append(cost)
    iterates.append(iterate[-1]) 
    onlineOptimizer.step(sample)
    

iterates = np.array(iterates) 
iterates2 = np.array(onlineOptimizer.iterates)
plt.figure()
plt.plot(np.log(costs),c='r')
plt.plot(np.log(onlineOptimizer.costs),c='b')

plt.figure()
for i in range(9): 
    plt.plot(iterates[:,i,0]) 

plt.figure()
for i in range(9): 
    plt.plot(iterates2[:,i,0]) 