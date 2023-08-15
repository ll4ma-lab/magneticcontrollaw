# -*- coding: utf-8 -*-
""" 
Created on Tue May 11 14:06:39 2021 
 
@author: Griffin Tabor 
"""
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np

import sys

sys.path.append("..")  # Adds higher directory to python modules path.
import matplotlib.pyplot as plt
import time
from PDController import PDController
from torch.autograd import Variable

from environments import PlanarProblem, SpaceProblem, SpaceEnvironments
import SplineProfile
from discreteTimeSimulator import DiscreteTimeSimulator
from ObjectProperties import compute_sphere_properties,ObjectProperties

from models import MultiMagModel
from solvers.modelOptimization import MultiMagSolver


env = SpaceProblem(SpaceEnvironments.AXIS)
env.magnets = env.magnets[:1]
controller = PDController(env.start_state, 0.005, 0.00001, 0.005, 0.00001, 0.0, 0.000)
velocityProfile = SplineProfile.SplineProfile(env, env.times)


radius = 0.02
sigma_copper = 5.87e07
mass, MOI = compute_sphere_properties(8940, radius)
object_properties = ObjectProperties(
    mass=mass, MOI=MOI, radius=radius, sigma=sigma_copper
)

if __name__ == "__main__":
    position = np.zeros((3, 1))
    model = MultiMagModel(env.magnets,object_properties,True)
    solver = MultiMagSolver(model, "all",object_properties, True)
    # np.random.seed(473)
    for i in range(1):

        desired_wrench = (np.random.random((6, 1)) - 0.5) * 1e-4
        s = time.time()

        control = solver.solve_wrench(position, desired_wrench)
        print(time.time() - s)
        if time.time() - s > 1 and i > 0:
            break


    from models import model_call_rotated_wrench
    import torch
    
    
    particles = 5
    
    
    zero = torch.zeros((particles))
    
    
    one = 1
    params = np.ones((particles,2))
    angles= torch.ones((2))
    
    
    model_vals = model.sphere_model.model_call(one,one,one,one)
    

    dipole_strength = 30 *torch.ones((1))
    rho = torch.ones((3,1))



    dist = 1
    radii = params[:,0]
    sigmas= params[:,1]

    #mass_matrix = self.construct_mass_matrix(object_properties, torch)
    frho0, trho0, frho90, fphi90, ttheta90 =  model.sphere_model.model_call(
        dist, 1, radii=radii, sigma=sigmas
    )
    
    mass_matrix_inv = np.ones((particles,6,6))
    rhos = rho.expand((3,particles))

    
    #model_vals = [torch.tensor(model_val) for model_val in model_vals]
    
    
    
    

    desired_a = torch.ones((6,1))
    
    
    
    
    # modeled_wrenches = model_call_rotated_wrench(
    #     dipole_strengths, angles_array, model_vals, rhos, library=torch, zero=zero
    # ).squeeze()
    


    from solvers.modelOptimization import SphericalAccelOptimization 
    
    
    class ExpectationControlOptimization(SphericalAccelOptimization):
        def __init__(self, weights):
            super().__init__(weights)
            self.mass_matrix_inv = torch.eye(6)
    
        def tensor_cost(self, x):
            
            selected_wrench = self.get_wrench(x)
            
            desired_matrix = self.desired.unsqueeze(1).expand((6,selected_wrench.shape[0],1))
            predicted_acceleration = torch.bmm(self.mass_matrix_inv,selected_wrench).transpose(0,1)
            error = 1e12 * (desired_matrix - predicted_acceleration)
            
            right =  (self.weights @ error.squeeze()).unsqueeze(2).transpose(0,1)
            left = error.transpose(0,1).transpose(1,2)
            
            cost = torch.bmm(left,right)
            return cost
    
        def tensor_error(self, x):
            #needs a sum across batch dimension
            selected_wrench = self.get_wrench(x)

            desired_matrix = self.desired.unsqueeze(1).expand((6,selected_wrench.shape[0],1))
            predicted_acceleration = torch.bmm(self.mass_matrix_inv,selected_wrench).transpose(0,1)
            error = 1e12 * (desired_matrix - predicted_acceleration)
            weighted_error = torch.sqrt(self.weights) @ error
            return weighted_error
        
        def get_wrench(self, x):
            particles = self.frho0.shape[0]
            
            dipole_strength = x[:1,0]
            angles = x[1:,0]

            dipole_strengths = dipole_strength.expand((particles))
            angles_array = angles.unsqueeze(1).expand((2,particles))
            
            rhos = rho.expand((3,particles))
            zeros = self.zero.expand(particles)
            return model_call_rotated_wrench(
                dipole_strengths,
                angles_array,
                (self.frho0, self.trho0, self.frho90, self.fphi90, self.ttheta90),
                rhos,
                library=torch,
                zero=zeros,
            )

    prob = ExpectationControlOptimization(np.eye(6))
    
    prob.frho0 = torch.tensor(frho0)
    prob.trho0 = torch.tensor(trho0)
    prob.frho90 = torch.tensor(frho90)
    prob.fphi90 = torch.tensor(fphi90)
    prob.ttheta90 = torch.tensor(ttheta90)
    
    prob.desired = torch.Tensor(desired_a)
    prob.rho = torch.tensor(rho)
    prob.mass_matrix_inv = torch.tensor(mass_matrix_inv)
    
    prob.cost(np.ones((3,1)))
    