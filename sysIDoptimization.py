#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 20:30:42 2021

@author: tabor
"""

import numpy as np
import time
import torch
from models import modelCallRotatedWrench,sphereModel
from ObjectProperties import ObjectProperties,buildMassMatrix,sphereProperties

from modelOptimization import buildQ,build_acceleration_Q
from ll4ma_opt.problems import Problem 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

from ll4ma_opt.solvers.line_search import NewtonMethod

class radiusSysid(Problem): 
    def __init__(self,weights,object_properties=None,boat = None): 
        super().__init__()
        
        
        self.singleModel = sphereModel(1)
        self.sigma_multiplier = 1e0
        
        self.error_multiplier = 1e6
        
        if object_properties:
            self.initial_solution = np.array([object_properties.radius,
                                              object_properties.sigma/self.sigma_multiplier]).reshape((-1,1))
        else:
            self.initial_solution = np.ones((2,1)) * 0.001
            self.initial_solution[1] = 5.8e7/self.sigma_multiplier
            
        
        self.min_bounds[0] = 1e-3
        self.max_bounds[0] = 5e-2

        self.min_bounds[1] = 2e5/self.sigma_multiplier
        self.max_bounds[1] = 2e8/self.sigma_multiplier
        

        self.weights =  self.make_tensor(weights)
        if boat:
            self.boat = boat.torch()
        else: 
            self.boat = boat

        
    def updateVals(self,controls,magnets,positions,linearVelocity,rotationalVelocity,drag_terms = (0,0)):
        #remove first control and position as we dont know its acceleration
        
        magNums = [control.mag_num for control in controls[1:]]
        chosenMagnetPose = np.array(magnets)[magNums]-positions[1:]
        dipoleStrength = [control.dipole_strength for control in controls[1:]]
        angles =  np.array([(control.psi,control.xi) for control in controls[1:]]).T
        timesteps = np.array([control.seconds for control in controls[1:]]).reshape((-1,1))

        self.n = len(controls[1:])
        
        
        rhos = -chosenMagnetPose.T
        dists=np.linalg.norm(chosenMagnetPose,axis=1)
        zero = np.array([0]*self.n)

        self.dipoleStrength = self.make_tensor(dipoleStrength)
        self.angles = self.make_tensor(angles)
        self.rhos =  self.make_tensor(rhos)
        self.dists = self.make_tensor(dists)
        
        self.zero =  self.make_tensor(zero)

        
        deltaV = np.hstack((linearVelocity[1:,:]-linearVelocity[:-1,:],rotationalVelocity[1:,:]-rotationalVelocity[:-1,:]))
        npA = deltaV /timesteps
        
        drag_force = drag_terms[0] * linearVelocity[:-1,:] + np.sign(linearVelocity[:-1,:]) * drag_terms[1] * linearVelocity[:-1,:]**2 
        
        self.drag_wrenches = self.make_tensor(np.hstack((-drag_force,np.zeros_like(drag_force))))
        self.accelerations = self.make_tensor(npA.T)
        
        self.materialDensity = self.make_tensor([8940.0])[0]
    def forward_acceleration(self, object_properties,dists,dipoleStrengths,angles,rhos,zero):
        
        mass_matrix = self.construct_mass_matrix(object_properties,torch)
        modelVals = self.singleModel.modelCall(dists,1,radii=object_properties.radius,sigma=object_properties.sigma)
        modelVals = [modelVal.squeeze() for modelVal in modelVals]
        modeled_wrenches = modelCallRotatedWrench(dipoleStrengths,angles,modelVals,rhos,library=torch,zero = zero).squeeze() 
        
        #print(modeled_wrenches)
        #print(self.drag_wrenches)
        wrenches = modeled_wrenches + self.drag_wrenches
        
        
        
        return torch.linalg.inv(mass_matrix)@ wrenches.T
    
    def get_object_properties(self, x, library=np):
        radius = x[0,0]
        sigma = x[1,0]
        mass,MOI = sphereProperties(self.materialDensity,radius,library)
        object_properties = ObjectProperties(mass=mass,MOI=MOI,radius=radius,sigma=sigma*self.sigma_multiplier)
        return object_properties

    def construct_mass_matrix(self, object_properties, library=np):
        if self.boat:
            object_properties = object_properties+self.boat
        M = buildMassMatrix(object_properties,library,False)
        return M
    
    def size(self):
        return 2
    
    def tensor_cost(self, x):
        object_properties = self.get_object_properties(x,torch)

        modeledAccelerations = self.forward_acceleration(object_properties,self.dists,self.dipoleStrength,self.angles,self.rhos,self.zero)
        error = (self.accelerations - modeledAccelerations)*self.error_multiplier
        
        
        left = (error.T @ self.weights).unsqueeze(1)
        right = error.T.unsqueeze(2)
        cost = torch.sum(torch.bmm(left,right))/ self.rhos.shape[0] #-  5e14* (x[0:1])**2

        return cost
    
    def tensor_error(self,x):
        object_properties = self.get_object_properties(x,torch)

        modeledAccelerations = self.forward_acceleration(object_properties,self.dists,self.dipoleStrength,self.angles,self.rhos,self.zero)
        error = (self.accelerations - modeledAccelerations)*self.error_multiplier
        
        
        weighted_error = (error.T @ torch.sqrt(self.weights/ self.rhos.shape[0]))
        return weighted_error.reshape(((-1,1)))
         
    
    
    def visualize_optimization(self, iterates, samples=50, func=np.abs, show_costs=False):
        fig = plt.figure()
        if show_costs:
            ax = fig.add_subplot(projection="3d")
        else:
            ax = fig.add_subplot()

        possible_xs = np.linspace(self.min_bounds[0], self.max_bounds[0], samples)
        possible_ys = np.linspace(self.min_bounds[1], self.max_bounds[1], samples)
        xs, ys = np.meshgrid(possible_xs, possible_ys)
        xs = xs.flatten()
        ys = ys.flatten()
        possibleCosts = [
            func(self.cost(np.stack((xs[i], ys[i])).reshape((2, 1)))) for i in range(len(xs))
        ]
        clip_value = 2 * np.median(possibleCosts)
        possibleCosts = np.array(possibleCosts)
        possibleCosts[possibleCosts > clip_value] = clip_value

        best_index = np.argmin(possibleCosts)
        print(
            "best brute force cost ",
            possibleCosts[best_index],
            " at ",
            xs[best_index],
            ys[best_index],
        )

        ax.set_xlim(self.min_bounds[0], self.max_bounds[0])
        ax.set_ylim(self.min_bounds[1], self.max_bounds[1])

        points = iterates[0].reshape((-1, 2))
        costs = [[min(func(self.cost(point.reshape((-1,1)))),clip_value)] for point in points]
        print(np.min(costs))

        if show_costs:
            ax.scatter(xs, ys, possibleCosts, color=(0, 0, 1, 0.5))
            #ax.set_zlim(0, clip_value)
            points_3d = np.concatenate((points, np.array(costs)), 1)
            # silly hack to get filled circle
            scat = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], linewidths=10, c="orange")
            scat2 = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], linewidths=10, c="orange")

            #scat._offsets3d = (points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
        else:
            scat = plt.scatter(points[:, 0], points[:, 1], linewidths=10, c="orange")
            scat.set_offsets(points)
            scat2 = None

        def update(num, scat, scat2, iterates, show_costs):
            points = iterates[num].reshape((-1, 2))
            costs = [[min(func(self.cost(point.reshape((-1,1)))),clip_value)] for point in points]
            points_3d = np.concatenate((points, np.array(costs)), 1)
            #print(points_3d.shape)
            if show_costs:
                scat._offsets3d = (points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
                scat2._offsets3d = (points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])

            else:
                scat.set_offsets(points)

        anim = FuncAnimation(
            fig,
            update,
            fargs=(scat, scat2, iterates, show_costs),
            frames=len(iterates),
            interval=100,
            repeat=True,
        )
        return anim

class SeperateRadiusSysid(radiusSysid):
    def __init__(self,weights,object_properties,boat = None): 
        super().__init__(weights=weights,boat=boat)
        
        if object_properties:
            self.initial_solution = np.array([object_properties.radius,
                                              object_properties.sigma/self.sigma_multiplier,
                                              object_properties.radius]).reshape((-1,1))
        else:
            self.initial_solution[2] = 0.1

        #radius and sigma keep the same
        
        #second radius for mass and MOI
        self.min_bounds[2] = 1e-5
        self.max_bounds[2] = 1e-0
    def size(self):
        return 3
    def get_object_properties(self, x, library=np):
        radius = x[0,0]
        sigma = x[1,0]
        mass_radius = x[2,0]
        mass,MOI = sphereProperties(self.materialDensity,mass_radius,library)
        object_properties = ObjectProperties(mass=mass,MOI=MOI,radius=radius,sigma=sigma*self.sigma_multiplier)
        return object_properties
    def visualize_optimization(self,iterates,func = np.abs): 
        fig,axs = plt.subplots(self.size(),1)
        
        for i in range(self.size()):
            axs[i].plot(iterates[:,i,0],'r')
        fig.suptitle('Iterates over time')
        
        fig,axs = plt.subplots(self.size(),1)
        fig.suptitle('Cost as iterate changes')

        for i in range(self.size()):
            min_value = self.min_bounds[i]
            max_value = self.max_bounds[i]
            xs = np.linspace(min_value,max_value,100)
                        
            ys = []
            for x in xs:
                iterate = np.copy(iterates[-1])
                iterate[i] = x
                ys.append(self.cost(iterate))
            
            x = iterates[-1][i]
            y = self.cost(iterates[-1])
            axs[i].plot([x],func([y]),'r',marker='o')
            axs[i].plot(xs,func(ys),'b')

class MassSysID(SeperateRadiusSysid):
    def __init__(self,weights,object_properties=None,boat = None): 
        super().__init__(weights=weights,object_properties=object_properties,boat=boat)
        
        if object_properties:
            self.initial_solution = np.array([object_properties.radius,
                                              object_properties.sigma/self.sigma_multiplier,
                                              object_properties.mass,
                                              object_properties.MOI[0,0],
                                              object_properties.MOI[1,1],
                                              object_properties.MOI[2,2]
                                              ]).reshape((-1,1))
        else:
            self.initial_solution = np.array([self.initial_solution[0,0],
                                              self.initial_solution[1,0],
                                              1.0, #mass
                                              1.0, #Ixx
                                              1.0, #Iyy
                                              1.0  #Izz
                                              ]).reshape((-1,1))
        
        self.min_bounds[0] = 1e-3
        
        self.min_bounds[1] = 1e4/self.sigma_multiplier
        
        #radius and sigma keep the same
        self.min_bounds[2] = 1e-10
        self.max_bounds[2] = 2
        
        self.min_bounds[3] = 1e-5 #mass
        self.min_bounds[3:] = 1e-10 #MOI
        
        self.max_bounds[3] = 2 #mass
        self.max_bounds[3:] = 1e0 #MOI
        
        
    def size(self):
        return 6
    
    def get_object_properties(self, x, library=np):
        radius = x[0,0]
        sigma = x[1,0]
        mass = x[2,0]
        MOI = library.diag(x[3:,0])
        object_properties = ObjectProperties(mass=mass,MOI=MOI,radius=radius,sigma=sigma*self.sigma_multiplier)
        
        return object_properties

 
    
    
        
def sysIDDemo(controls,magnets,positions,linearVelocity,rotationalVelocity,sysID_type = radiusSysid, boat=None):

    optimizerParams = {'alpha':1,'rho':0.9,'c1':1e-9,'FP_PRECISION': 1e-8,'min_alpha': 1e-20}
    prob = sysID_type(build_acceleration_Q('all'),boat) # built with M(0.04) Q M(0.04)
    optimizer = NewtonMethod(problem=prob,**optimizerParams)
    prob.updateVals(controls,magnets,positions,linearVelocity,rotationalVelocity)
    result = optimizer.optimize(prob.initial_solution,max_iterations=50)
    prob.visualize_optimization(result.iterates)
    return result