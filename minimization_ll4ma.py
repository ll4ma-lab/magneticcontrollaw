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

class demo(Problem): 
    def __init__(self): 
        super().__init__()
        self.initial_solution = np.ones((1,1)) * 0.001 
    def forward(self,x): 
        clippedTheta = torch.sigmoid(x) * 360 - 180 
        return clippedTheta 
 
    def loss_fn(self,x): 
        error = 0.7-torch.cos(x*np.pi/180.0) 
        return 1e10 * error*error  
     
    def tensor_cost(self, x): 
        return self.loss_fn(self.forward(x)) 
    def visualize_optimization(self,iterates): 
        angles = self.forward(torch.Tensor(iterates.reshape([-1]))) 
        plt.plot(angles) 
         
def optimize(prob): 
    solver = NewtonMethod(alpha, rho, prob) 
    solution, cost, converged, iterates = solver.optimize(prob.initial_solution, max_iterations=max_iterations) 
    return  (solution, cost, converged, iterates) 
 
env = SpaceProblem(4)
controller = PDController(env.startState,0.005,0.00001,0.005,0.00001,0.0,0.000)
velocityProfile = SplineProfile.SplineProfile(env,np.sum(env.times),env.times)
simulator = discreteTimeSimulator(env.startState,1,sphereProperties(8940,0.04)) #density, diamter

desiredState = velocityProfile.getDesiredState(1,simulator.state)
desired_wrench,p,d = controller.getDesiredWrench(simulator.state,desiredState)

alpha = 1.0 
rho = 0.5 
max_iterations = 10 
prob = demo()
solver = NewtonMethod(alpha, rho, prob) 
solution, cost, converged, iterates = solver.optimize(prob.initial_solution, max_iterations=max_iterations) 
model = cylinderModel(env.magnets,0.04)


strength  = Variable(torch.DoubleTensor([0.0]), requires_grad=True)

model.eddyFTRotated(0.01,strength,15,0.25,use_pytorch=True)






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
