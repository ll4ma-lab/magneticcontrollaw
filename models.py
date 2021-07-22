#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 17:23:35 2021

@author: tabor
"""
import numpy as np
import torch

from problems.problem import Problem 
import matplotlib.pyplot as plt 
torch.set_default_tensor_type(torch.DoubleTensor)
import torch.multiprocessing  as multiprocessing

from solvers.line_search import NewtonMethod 
import time
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
        error = 1e10*(self.desiredWrench - selectedWrench)
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
     
        
     
optimizerParams = {'alpha':1.0,'ro':0.9,'c1':1e-9,'FP_PRECISION': 1e-4,'min_alpha': 1e-10}
def optimizeSingle(prob): 
    solver = NewtonMethod(problem=prob,**optimizerParams) 
    solution, cost, converged, iterates = solver.optimize(prob.initial_solution)  
    return solution, cost, converged, iterates


def buildQ(stateCostString):
    letterOptions = ["x","y","z","r","p","t"]
    activated = [0]*6
    if("all" in stateCostString):
        stateCostString = letterOptions
    for index in range(len(letterOptions)):
        if(letterOptions[index] in stateCostString):
            #print(index)
            if index < 3:   
                activated[index] = 1.0 #/ (objectProperties[0] * tolerances[0])
            else: 
                activated[index] = 200.0 #/ (objectProperties[1][0,0] * tolerances[1])

    Q = np.diag(activated)
    print(Q)
    return Q

class multiMagSolver():
    def __init__(self,model,stateCostString,numProcesses=8):
        self.model = model
        self.problems = []
        self.time = 0
        Qweights = buildQ(stateCostString)
        for i in range(3*len(model.magnets)):
            prob = cylinderOptimization(Qweights)
            prob.rhoMode = (i%3) ==2
            self.problems.append(prob)
        ctx = multiprocessing.get_context('spawn')
        self.pool =  ctx.Pool(processes=numProcesses)
        

        self.optimizeSingle = lambda prob: NewtonMethod(self.multiprocessing).optimize(prob.initial_solution)   
        self.iterations = []
        self.times = []
    def modifyProblems(self,desiredWrench,position):
        relativePoses = (np.array(self.model.magnets) - np.transpose(position)).tolist()
        for i in range(len(self.model.magnets)):
            chosenMagnetPose = relativePoses[i]
            omniFrame = getOmniFrame(chosenMagnetPose)
            dist = np.linalg.norm(chosenMagnetPose)
            wrenches = self.model.model.modelCall(dist,1) #1 dipole strength
            for j in range(3):
                self.problems[i*3+j].frame = torch.Tensor(omniFrame)
                self.problems[i*3+j].desiredWrench = torch.Tensor(desiredWrench)
                self.problems[i*3+j].unitStrengthWrench = torch.tensor(wrenches[j])
    def solveWrench(self,position,desiredWrench):
        s = time.time()
        self.modifyProblems(desiredWrench, position)
        results = self.pool.map(optimizeSingle,self.problems)
        iterations = [result[3].shape[0] for result in results]
        self.iterations.append((iterations))
        noActionCost  = self.problems[0].cost(np.zeros((2,1)))
        costs = [result[1] for result in results]
        costs.append(noActionCost)
        
        bestAction = np.argmin(costs)
        if(bestAction == len(costs)-1):
            control = [0,0,0,0]
            print('shit')
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


def getOmniFrame(pos):
    Z = np.array([0, 0, 1])

    dist = np.linalg.norm(pos)

    omni_Z = -1*(pos/dist);
    
    omni_Y = np.cross(omni_Z,Z)
    omni_Y = omni_Y/np.sqrt(np.dot(omni_Y,omni_Y)) #unit vectorise
    
    omni_X = np.cross(omni_Y,omni_Z);
    # Combine to build rotation matrix
    omniFrame = np.transpose(np.array([omni_X, omni_Y, omni_Z]));
    return omniFrame
def averageDict(a,b):
    newDict = {}
    for key in a.keys():
        newDict[key] = ((np.array(a[key])+np.array(b[key]))/2).tolist()
    return newDict

def rotZ(yaw,library = np):
    yaw = yaw*np.pi/180.0
    one,zero = (1.0,0.0) if library is np else (torch.tensor([1.0]),torch.tensor([0.0]))
    
    RZ = library.stack([
                library.stack([library.cos(yaw), -library.sin(yaw), zero]),
                library.stack([library.sin(yaw), library.cos(yaw), zero]),
                library.stack([zero, zero, one])]).reshape(3,3)
    return RZ
class multiMagModel():
    def __init__(self,magnets,diameter):
        self.time = 0
        self.magnets = magnets
        self.diameter = diameter
        self.model = cylinderModel(diameter)
    def forwardPass(self,position,control):
        #control[0] = magnetNumber
        #control[1] = modelNumber
        #control[2] = power
        #control[3] = theta
        s = time.time()
        chosenMagnetPose = (np.array(self.magnets) - np.transpose(position)).tolist()[control[0]]
        omniFrame = getOmniFrame(chosenMagnetPose)
        
        if(control[1] == 2):
            rotation = np.dot(omniFrame,rotZ(control[3]))
        else:
            rotation = omniFrame
        
        doubleRotMatrix = np.zeros((6,6))
        doubleRotMatrix[:3,:3] = rotation
        doubleRotMatrix[3:,3:] = rotation
        
        wrench = self.model.modelCall(np.linalg.norm(chosenMagnetPose),control[2])[control[1]]
        
        rotatedWrench = doubleRotMatrix@wrench
        self.time += time.time()-s
        return rotatedWrench

class cylinderModel():
    def __init__(self,diameter):
        self.time = 0.0
        sigma_copper = 5.8E+07;
        sigma_al = 3.77E+07;
        self.sigma = sigma_copper;
        self.do = diameter;
        self.radii = diameter/2;
        self.freq = 15
        self.mu = 4*np.pi*1e-7;  
        
        self.feaConstants = {}
        self.feaConstants['fzz'] = [430,2.95,-0.101,-9.26,7]
        self.feaConstants['tzz'] = [6840,3.00,-0.0986,-13.2,6]
        self.feaConstants['frhorho'] = [266,2.60,-0.101,-7.65,7]
        self.feaConstants['frhophi'] = [6040,3.45,-0.102,-14.3,7]
        self.feaConstants['trhoz'] = [8100,3.60,-0.0985,-15.7,6]
        
        
        self.expConstants = {}
        self.expConstants['fzz'] = [467,2.81,-0.0969,-9.75,7]
        self.expConstants['tzz'] = [6900,3.35,-0.0990,-14.9,6]
        self.expConstants['frhorho'] = [282,3.20,-0.0980,-9.41,7]
        self.expConstants['frhophi'] = [5870,3.49,-0.0973,-14.6,7]
        self.expConstants['trhoz'] = [8000,3.40,-0.0928,-15.0,6]
        
        self.constants = averageDict(self.feaConstants,self.expConstants)

    def modelEquation(self,constants,distance,dipoleStrength,p0RPower):
        #P0 has r^4 for forces and r^3 for torques
        P1 = self.sigma*self.mu*self.freq*self.radii*self.radii
        P2 = distance/self.radii

        numeratorBase = constants[0]*P1
        numeratorPower = constants[1]*(P1**constants[2])
        numerator = (numeratorBase**numeratorPower) * (10**constants[3])
        
        denominator = P2**constants[4]
        P0 = numerator/denominator

        c=  self.mu*(dipoleStrength**2)*(self.radii**-p0RPower)

        wrench = P0*c
        
        return wrench
        
    def modelCall(self,dist,dipoleStrength):
        fzz = self.modelEquation(self.constants['fzz'],dist,dipoleStrength,4)
        tzz = self.modelEquation(self.constants['tzz'],dist,dipoleStrength,3)
        
        frhorho = self.modelEquation(self.constants['frhorho'],dist,dipoleStrength,4)
        frhophi = self.modelEquation(self.constants['frhophi'],dist,dipoleStrength,4)
        trhoz = self.modelEquation(self.constants['trhoz'],dist,dipoleStrength,3)
        
        pointingModel = np.array([0,0,fzz,0,0,tzz]).reshape((6,1))
        negPointingModel = np.array([0,0,fzz,0,0,-tzz]).reshape((6,1))
        
        rhoModel = np.array([0,-frhophi,frhorho,-trhoz,0,0]).reshape((6,1))
        return pointingModel,negPointingModel,rhoModel



