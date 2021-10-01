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

def getCost(decisionVariable): 
    global prob
    return prob.cost(decisionVariable)


env = SpaceProblem(4)
controller = PDController(env.startState,0.005,0.00001,0.005,0.00001,0.0,0.000)
velocityProfile = SplineProfile.SplineProfile(env,np.sum(env.times),env.times)


diameter = 0.04
objectProperties = sphereProperties(8940,diameter)



optimizerParams = {'alpha':1.0,'ro':0.9,'c1':1e-9,'FP_PRECISION': 1e-2,'min_alpha': 1e-15}



newModel = multiMagModel(env.magnets,0.04)
#control = [1,2,np.random.random()*40,np.random.random()*360-180]
control = [5, 2, 38, 173]
print(newModel.forwardPass(np.zeros((3,1)), control))
#np.random.seed(473)

#desired_wrench = (np.random.random((6,1))-0.5)*1e-4    
#desired_wrench = (np.random.random((6,1))-0.5)*1e-7
#desired_wrench = (np.random.random((6,1))-0.5)*1e-7
#desired_wrench = (np.random.random((6,1))-0.5)*1e-7
desired_wrench = (np.random.random((6,1))-0.5)*1e-4


results = []


def fixmaxDegree(angles):
    return (angles+180)%360 - 180

BRUTEFORCECOLOR = False
if __name__ == '__main__':
    builder = multiMagSolver(newModel,'all',True)

    for _ in range(100):
        desired_wrench = (np.random.random((6,1))-0.5)*1e-4
        desired_wrench[3:,0]*=1e-1

        builder.modifyProblems(desired_wrench,np.zeros((3,1)))
    
        for i in range(len(builder.problems[:1])):
            prob = builder.problems[control[0]]
            
        
        s = time.time()
        optimizer = NewtonMethod(problem=prob,**optimizerParams) 
        solution, cost, converged, iterates = optimizer.optimize(np.array([20,0,0]).reshape((3,1))) 
        solution2, cost2, converged2, iterates2 = optimizer.optimize(np.array([20,180,180]).reshape((3,1))) 
        print('elapsed newton time ',time.time()-s)
    
        #print(solution,cost,converged,iterates.shape)
    
        strength = np.linspace(0,40,21)
        psis = np.linspace(-180,180,91)
        xis = np.linspace(-180,180,91)
        xx, yy, zz = np.meshgrid(strength, psis,xis)
        x=xx.reshape([-1])
        y=yy.reshape([-1])
        z=zz.reshape([-1])
        
        decisionVariablesList = []
        
        
        func = np.abs
        
        
        for i in range(x.shape[0]):
            decisionVariables = np.array([x[i],y[i],z[i]]).reshape((3,1))
            decisionVariablesList.append(decisionVariables)
        
        ctx = multiprocessing.get_context('spawn')
    
        costs = [getCost(decisionVariable) for decisionVariable in decisionVariablesList]
            
        
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
        
        print('bruteForce best ',x[bestarg],y[bestarg],z[bestarg],' with cost ',costs[bestarg])
        print('newton method best ',solution[0],solution[1],solution[2],'with cost ', cost,' ratio ', cost/costs[bestarg])
        print('newton method best ',solution2[0],solution2[1],solution2[2],'with cost ', cost2,' ratio ', cost2/costs[bestarg])
        if(cost2/costs[bestarg] >1 and cost/costs[bestarg] >1):
            for params in [(90,90),(45,45),(45,0),(0,45),(90,0),(0,90),(180,0),(0,180)]:
                solution3, cost3, converged3, iterates3 = optimizer.optimize(np.array([20,params[0],params[1]]).reshape((3,1))) 
                print(params,cost3/costs[bestarg])

            break
    
        plot = False    
        
        if plot:
            foundCosts = []
            for j in range(iterates.shape[0]):
                foundCosts.append(func(prob.cost(iterates[j,:,:])))
            foundCosts2 = []
            for j in range(iterates2.shape[0]):
                foundCosts2.append(func(prob.cost(iterates2[j,:,:])))
            
            npCosts = np.array(costs)
            npc = np.array(colors)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            
            for streng in strength:
                indices = np.argwhere(x==streng)[:,0].tolist()
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
        
                fig.suptitle(str(streng) +' dipole', fontsize=16)
        
                ax.scatter(y[indices],z[indices],npCosts[indices],s=1,c=npc[indices])
                ax.plot(iterates[:,1,0],iterates[:,2,0],foundCosts,c='r',markersize=200,linewidth=6)
                ax.plot(fixmaxDegree(iterates2[:,1,0]),fixmaxDegree(iterates2[:,2,0]),foundCosts2,c='g',markersize=200,linewidth=6)
    
                plt.show()
    
        #print('elapsed time ',time.time()-s)
        
        producedControl = [0,solution[0,0],solution[1,0],solution[2,0]]
        bestWrench = newModel.forwardPass(np.zeros((3,1)), producedControl,True)
        #print(bestWrench)
        #print(desired_wrench)
    
        producedControl2 = [0,solution2[0,0],solution2[1,0],solution2[2,0]]
        bestWrench2 = newModel.forwardPass(np.zeros((3,1)), producedControl2,True)
        #print(bestWrench2)