# -*- coding: utf-8 -*- 
""" 
Created on Tue May 11 14:06:39 2021 
 
@author: Griffin Tabor 
""" 
from problems.problem import Problem 
 
from multiprocessing import Pool, TimeoutError 
 
import numpy as np 
import torch 
from solvers.line_search import GradientDescent, NewtonMethod, NewtonMethodWolfeLine 
import matplotlib.pyplot as plt 
import time 
from PDController import PDController
from torch.autograd import Variable

from cylinderModel import cylinderSolver,cylinderModel
from environments import PlanarProblem,SpaceProblem
import SplineProfile
from discreteTimeSimulator import discreteTimeSimulator,sphereProperties


from models import cylinderModel as cylinderModel2
from models import multiMagModel,demo,problemBuilder



def optimize(prob): 
    solver = NewtonMethod(alpha, rho, prob) 
    solution, cost, converged, iterates = solver.optimize(prob.initial_solution, max_iterations=max_iterations) 
    return  (solution, cost, converged, iterates) 
 
env = SpaceProblem(4)
controller = PDController(env.startState,0.005,0.00001,0.005,0.00001,0.0,0.000)
velocityProfile = SplineProfile.SplineProfile(env,np.sum(env.times),env.times)
simulator = discreteTimeSimulator(env.startState,1,sphereProperties(8940,0.04)) #density, diamter


diameter = 0.04
objectProperties = sphereProperties(8940,diameter)
model = cylinderModel(env.magnets,diameter)
solver = cylinderSolver(env.magnets)
solver.setup(model,env.trajectoryDimensions,objectProperties,env.tolerances)



desiredState = velocityProfile.getDesiredState(1,simulator.state)
desired_wrench,p,d = controller.getDesiredWrench(simulator.state,desiredState)

alpha = 1.0 
rho = 0.9
max_iterations = 100


newModel = multiMagModel(env.magnets*2,0.04)
builder = problemBuilder(newModel)
#desired_wrench = np.array([1,0,0,0,0,0]).reshape((6,1))
control = [0,2,np.random.random()*40,np.random.random()*360-180]
control = [0, 2, 38.36714973571669, 172.5982599543529]
desired_wrench = newModel.forwardPass(simulator.state[0][:3,3],control)

oldPredictedControl,oldIterates =  solver.solveWrench(model,simulator.state[0][:3,3],desired_wrench,env.trajectoryDimensions)
formatedOldControl = np.array([oldPredictedControl[2],oldPredictedControl[3]]).reshape((2,1))
oldIterates = np.array(oldIterates)

builder.modifyProblems(desired_wrench,simulator.state[0][:3,3])
for i in range(len(builder.problems[:1])):
    prob = builder.problems[2]
    optimizer = NewtonMethod(alpha, rho, prob,FP_PRECISION=1e-20) 
    solution, cost, converged, iterates = optimizer.optimize(prob.initial_solution, max_iterations=max_iterations) 
    print(solution,cost,converged,iterates.shape)


strength = np.linspace(0,40,160)
angles = np.linspace(-360,360,400)
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


def optimize(decisionVariable): 
    solver = NewtonMethod(alpha, rho, prob) 
    solution, cost, converged, iterates = solver.optimize(decisionVariable, max_iterations=max_iterations)
    if(cost<1e-5):
        return 'b'
    else:
        return 'y'
with Pool(processes=8) as pool: 
    colors=pool.map(optimize, decisionVariablesList)   
    
    

    
bestarg = np.argmin(costs)

print('bruteForce best ',x[bestarg],y[bestarg],' with cost ',costs[bestarg])
print('newton method best ',solution[0],solution[1],'with cost ', cost)
print('old best ',oldPredictedControl[2:], ' with cost ',prob.cost(formatedOldControl))

foundCosts = []
for j in range(iterates.shape[0]):
    foundCosts.append(func(prob.cost(iterates[j,:,:])))

foundOldCosts = []
for j in range(oldIterates.shape[0]):
    formatedOldDecision = np.array([oldIterates[j][0],oldIterates[j][1]]).reshape((2,1))
    foundOldCosts.append(func(prob.cost(formatedOldDecision)))
    
    
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x,y,costs,s=1,c=colors)
ax.plot(np.abs(iterates[:,0,0]),iterates[:,1,0],foundCosts,c='r',markersize=200,linewidth=6)
ax.plot(np.abs(oldIterates[:,0]),oldIterates[:,1],foundOldCosts,c='g',markersize=200,linewidth=6)
plt.show()






#dist =0.01
#strength =5
#
#control = [0,0,1,45]
#wrench = model.forwardPass(simulator.state[0][:3,3],control)
#print(wrench)
#if(control[1]==0 ):
#    control[1] = 2
#else:
#    control[1] = control[1] -1
#
#newModel = multiMagModel(env.magnets,0.04)
#newWrench = newModel.forwardPass(simulator.state[0][:3,3],control)
#print(newWrench)

# start = time.time() 
# 
#probs = [demo() for i  in range(10)] 
#print('prob creation ',time.time()-start) 
#def optimize(prob): 
#    solver = NewtonMethod(alpha, rho, prob) 
#    solution, cost, converged, iterates = solver.optimize(prob.initial_solution, max_iterations=max_iterations) 
#    return  (solution, cost, converged, iterates) 
 
#if __name__ == '__main__': 
#    start = time.time() 
#    with Pool(processes=4) as pool: 
#        print('pool creation ',time.time()-start) 
# 
#        middle = time.time() 
#        a=pool.map(optimize, probs) 
#        print('first time ',time.time()-middle) 
#         
#        middle = time.time() 
#        a=pool.map(optimize, probs) 
#        print('second time ', time.time()-middle) 
#        print('all ',time.time()-start) 
#
#    print('all + free ',time.time()-start) 
