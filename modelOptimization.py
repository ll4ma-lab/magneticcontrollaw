#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:28:31 2021

@author: tabor
"""
import numpy as np
import torch


import time
from problems.problem import Problem 
import matplotlib.pyplot as plt 
torch.set_default_tensor_type(torch.DoubleTensor)
import torch.multiprocessing  as multiprocessing

from solvers.line_search import NewtonMethod 

from models import rotZ,modelCallRotatedWrench,getOmniFrame

optimizerParams = {'alpha':1.0,'ro':0.5,'c1':1e-9,'FP_PRECISION': 1e-15,'min_alpha': 1e-15}
def optimizeSingle(prob):
    solver = NewtonMethod(problem=prob,**optimizerParams) 
    solution, cost, converged, iterates = solver.optimize(prob.initial_solution,max_iterations=25)  
    return solution, cost, converged, iterates


def buildQ(stateCostString):
    letterOptions = ["x","y","z","r","p","t"]
    activated = [0]*6
    if("all" in stateCostString):
        stateCostString = letterOptions
    for index in range(len(letterOptions)):
        if(letterOptions[index] in stateCostString):
            if index < 3:   
                activated[index] = 1.0 #/ (objectProperties[0] * tolerances[0])
            else: 
                activated[index] = 200.0 #/ (objectProperties[1][0,0] * tolerances[1])

    Q = np.diag(activated)
    return Q

class cylinderOptimization(Problem): 
    def __init__(self,weights): 
        super().__init__()
        self.initial_solution = np.array([20,0]).reshape((2,1))
        self.boundConstraints[0] = (0.1,40)
        
        self.desiredWrench = torch.zeros((6,1))
        self.unitStrengthWrench = torch.zeros((6,1))
        self.frame = torch.eye(3)
        self.rhoMode = False
        self.weights = torch.tensor(weights)
        
    def pointingModelLoss(self,x,frame):
        selectedWrench = x[0]**2 * torch.cat((frame@self.unitStrengthWrench[:3],frame@self.unitStrengthWrench[3:]))
        error = 1e13*(self.desiredWrench - selectedWrench)
        cost = error.t()@self.weights@error
        return cost    
    def rhoLoss(self,x):
        frame = torch.mm(self.frame,rotZ(x[1],torch))
        return self.pointingModelLoss(x,frame)
    def tensor_cost(self, x): 
        if self.rhoMode:
            return self.rhoLoss(x)
        else:
            return self.pointingModelLoss(x,self.frame)
    def visualize_optimization(self,iterates): 
        plt.plot(iterates[:,0,0],iterates[:,1,0]) 
class sphericalOptimization(Problem):
    def __init__(self,weights): 
        super().__init__()
        self.initial_solution = np.array([20,0,0]).reshape((3,1))
        self.boundConstraints[0] = (0.1,40)
        
        self.desiredWrench = torch.zeros((6,1))

        self.weights = torch.tensor(weights)
        
        self.frho0 = torch.tensor(0.0)
        self.trho0 = torch.tensor(0.0)
        self.frho90 = torch.tensor(0.0)
        self.fphi90 = torch.tensor(0.0)
        self.ttheta90  = torch.tensor(0.0)
        
        self.rho = torch.tensor(np.zeros((3,1)))
        
        self.zero = torch.zeros(1)
    def tensor_cost(self, x): 
        selectedWrench = modelCallRotatedWrench(x[0],x[1:],(self.frho0,self.trho0,self.frho90,self.fphi90,self.ttheta90),self.rho,library=torch,zero = self.zero)
        error = 1e10*(self.desiredWrench - selectedWrench)[0]
        cost = error.t()@self.weights@error
        return cost
    
    def visualize_optimization(self,iterates): 
        plt.plot(iterates[:,0,0],iterates[:,1,0]) 


def getCost(control):
    return control[0].cost(control[1])


class multiMagSolver():
    def __init__(self,model,stateCostString,sphereModel = False,numProcesses=8):
        self.model = model
        self.problems = []
        self.time = 0
        self.useSphereModel = sphereModel
        self.costs = []
        Qweights = buildQ(stateCostString)
        if(sphereModel):
            for i in range(2*len(model.magnets)):
                prob = sphericalOptimization(Qweights)
                if i%2 == 0:
                    prob.initial_solution = np.array([20,0,0]).reshape((3,1))
                if i%2 == 1:
                    prob.initial_solution = np.array([20,180,180]).reshape((3,1))
                self.problems.append(prob)
        else:     
            for i in range(3*len(model.magnets)):
                prob = cylinderOptimization(Qweights)
                prob.rhoMode = (i%3) ==2
                self.problems.append(prob)
        ctx = multiprocessing.get_context('spawn')
        self.pool =  ctx.Pool(processes=numProcesses)
        
        self.iterations = []
        self.times = []
    def modifyProblems(self,desiredWrench,position):
        if self.useSphereModel:
            self.modifySphereProblems(desiredWrench,position)
        else: 
            self.modifyCylinderProblems(desiredWrench, position)
    def modifySphereProblems(self,desiredWrench,position):
        relativePoses = np.array(self.model.magnets) - np.transpose(position)
        for i in range(len(self.model.magnets)):
            chosenMagnetPose = relativePoses[i]
            rho = (-1*chosenMagnetPose).reshape((3,1))
            dist = np.linalg.norm(chosenMagnetPose)
            frho0,trho0,frho90,fphi90,ttheta90 = self.model.sphereModel.modelCall(dist,1)
            for j in range(2):
                self.problems[i*2+j].frho0 = torch.tensor(frho0)
                self.problems[i*2+j].trho0 = torch.tensor(trho0)
                self.problems[i*2+j].frho90 = torch.tensor(frho90)
                self.problems[i*2+j].fphi90 = torch.tensor(fphi90)
                self.problems[i*2+j].ttheta90 = torch.tensor(ttheta90)
                
                self.problems[i*2+j].desiredWrench = torch.Tensor(desiredWrench)
                self.problems[i*2+j].rho = torch.tensor(rho)

    def modifyCylinderProblems(self,desiredWrench,position):
        relativePoses = (np.array(self.model.magnets) - np.transpose(position)).tolist()

        for i in range(len(self.model.magnets)):
            chosenMagnetPose = relativePoses[i]
            omniFrame = getOmniFrame(chosenMagnetPose)
            dist = np.linalg.norm(chosenMagnetPose)
            wrenches = self.model.cylinderModel.modelCall(dist,1) #1 dipole strength
            for j in range(3):
                self.problems[i*3+j].frame = torch.Tensor(omniFrame)
                self.problems[i*3+j].desiredWrench = torch.Tensor(desiredWrench)
                self.problems[i*3+j].unitStrengthWrench = torch.tensor(wrenches[j])
    def solveWrenchCylinder(self,position,desiredWrench):
        s = time.time()
        self.modifyProblems(desiredWrench, position)
        results = self.pool.map(optimizeSingle,self.problems)
        iterations = [result[3].shape[0] for result in results]
        self.iterations.append((iterations))
        noActionCost  = self.problems[0].cost(np.zeros((2,1)))
        costs = [result[1] for result in results]
        self.costs.append(costs)
        bestAction = np.argmin(costs)

        if(costs[bestAction]>noActionCost):
            control = [0,0,0,0]
            #print('shit')
        else:
            magNum = bestAction//3
            modelNum = bestAction%3
            strength = results[bestAction][0][0,0]
            theta = results[bestAction][0][1,0]
            control = [magNum,modelNum,strength,theta]
            
        timestep = time.time()-s
        self.times.append(timestep)
        self.time += timestep
        return control
    def solveWrench(self,position,desiredWrench):
        if(self.useSphereModel):
            return self.solveWrenchSphere(position,desiredWrench)
        else:
            return self.solveWrenchCylinder(position,desiredWrench)
    def solveWrenchSphere(self,position,desiredWrench):
        s = time.time()
        self.modifyProblems(desiredWrench, position)
        #results = [optimizeSingle(prob) for prob in self.problems]
        results = self.pool.map(optimizeSingle,self.problems)
        iterations = [result[3].shape[0] for result in results]
        self.iterations.append((iterations))
        noActionCost  = self.problems[0].cost(np.zeros((3,1)))
        costs = [result[1] for result in results]
        self.costs.append(costs)

        bestAction = np.argmin(costs)

        if(costs[bestAction]>noActionCost):
            control = [0,0,0,0]
            #print('shit')
        else:
            magNum = bestAction//2
            strength = results[bestAction][0][0,0]
            psi = results[bestAction][0][1,0]
            xi = results[bestAction][0][2,0]
            control = [magNum,strength,psi,xi]
            
        timestep = time.time()-s
        self.times.append(timestep)
        self.time += timestep
        return control
    def bruteSphereSolver(self,position,desiredWrench):
        self.modifyProblems(desiredWrench, position)

        s = time.time()
        controls = []
        for mag in range(len(self.model.magnets)):
            for i in range(-180,180,3):
                for j in range(-180,180,3):
                    for k in range(0,40):
                        control = (self.problems[mag*2],np.array((k,i,j)).reshape((3,1)),mag)
                        controls.append(control)
                        
        results = self.pool.map(getCost,controls)
        index = np.argmin(results)
        print('brute time ' , time.time()-s)
        selected = controls[index]
        control = [selected[2],selected[1][0,0],selected[1][1,0],selected[1][2,0]]
        
        
        return control
    
    