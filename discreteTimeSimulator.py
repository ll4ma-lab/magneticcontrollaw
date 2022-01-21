# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:40:03 2021

@author: Griffin Tabor
"""
import numpy as np
import time
from se3Utils import buildSkew,repairRotMatrix
from tqdm import tqdm



def buildMassMatrix(object_properties,library=np,torqueTop=True):
    mass = object_properties['mass']
    MOI = object_properties['MOI']

    massComponent = library.eye(3) *mass
    offDiagonal = library.zeros((3,3))
    
    if torqueTop:
        top = library.vstack((MOI,offDiagonal))    
        bottom = library.vstack((offDiagonal,massComponent))
    else:
        top = library.vstack((massComponent,offDiagonal))    
        bottom = library.vstack((offDiagonal,MOI))
    full = library.vstack((top.T,bottom.T)).T
    return full

def state_noise(state):
    
    transform = np.copy(state[0])
 
    current_linear_velocity = transform[:3,:3] @ state[2]
    current_rotational_velocity = transform[:3,:3] @ state[1]
    
    
    position_noise = 0#np.random.normal(0,3e-4,(3)) 
    linear_velocity_noise = 0#np.random.normal(0,3e-5,(3,1))
    rotational_velocity_noise = 0#np.random.normal(0,1e-5,(3,1))
    
    transform[:3,3] += position_noise
    current_linear_velocity += linear_velocity_noise
    current_rotational_velocity += rotational_velocity_noise

    tf_inverse = np.linalg.inv(transform)[:3,:3]
    
    state = [transform,tf_inverse @ current_rotational_velocity, tf_inverse @ current_linear_velocity]
    return state
    
    
        
    


def simulateExperiment(simulator,controller,env,velocityProfile,model,solver,adaptive,numSteps=10,dt=1,cheat=False,debugPrints = True,runAdaptive= False):
    state = simulator.state
    states = []
    wrenches = []
    desiredWrenchs = []
    desiredStates = []
    controls = []
    start = time.time()
    minimizationCosts = []


    iterates = range(1,numSteps+1)
    if(debugPrints):
        iterates = tqdm(iterates)
    
    for i in iterates:
        if i == 300 and runAdaptive:
            model.cylinderModel.object_properties['radius']*=1.5
            #arbitrarily change radius without updating mass
        desiredState = velocityProfile.getDesiredState(i*dt,state)
        desired_wrench,p,d = controller.getDesiredWrench(state,desiredState)
       
        desiredStates.append(desiredState)
        desiredWrenchs.append(desired_wrench)
        if(cheat):
            trueWrench = desired_wrench
            control = [0,0,0,0]
        else:
            control =  solver.solveWrench(state[0][:3,3],desired_wrench)
            trueWrench = model.forwardPass(state[0][:3,3],control)
            wrenchError = trueWrench-desired_wrench
            cost = np.sum((wrenchError[:3,:]**2)) + 200*np.sum((wrenchError[3:,:]**2))
            minimizationCosts.append(cost)
        wrenches.append(trueWrench)
        controls.append(control)
       
        env.problem_specific_modification(state,trueWrench)
        if runAdaptive:
            adaptive.step(state_noise(state),control)
        states.append(state)
        state = simulator.simulate(trueWrench)

    
    
    if debugPrints:
        runTime = time.time() - start
        print(runTime)

    return states,wrenches,desiredWrenchs,desiredStates,controls,minimizationCosts


class discreteTimeSimulator:
    def __init__(self,state,timestep,object_properties):
        self.t = timestep
        self.dt = 0.002
            
        from scipy.spatial.transform import Rotation as R
        
        worldToBodyRotation = R.from_euler('xyz',state[3:6,0])
        worldToBodyR = worldToBodyRotation.as_matrix()
        
        
        bodyToWorldR = np.transpose(worldToBodyR)
        bodyToWorld = np.eye(4)
        bodyToWorld[:3,:3] = bodyToWorldR
        bodyToWorld[:3,3] = state[:3,0]


        Ωsc = np.dot(worldToBodyR,state[9:12,:])
        Vsc = np.dot(worldToBodyR,state[6:9,:])
        self.state = (bodyToWorld,Ωsc,Vsc)

        self.Isc = buildMassMatrix(object_properties)
        self.steps = int(self.t/self.dt)
        
        self.time = 0

    def simulate(self,control):
        self.state = self.forwardPass(self.state,control)
        return self.state
    def forwardPass(self,state,FT):
        start = time.time()

        #Notation and math from https://downloads.hindawi.com/journals/ijae/2017/4034328.pdf
        (bodyToWorld0,Ωsc0,Vsc0) = state
        
        bodyToWorld = np.copy(bodyToWorld0)
        Ωsc = np.copy(Ωsc0)
        Vsc = np.copy(Vsc0)
        
        FT = np.reshape(FT,(6,1))
        
        for i in range(self.steps):
            rotrot = np.concatenate((np.concatenate((np.transpose(bodyToWorld[:3,:3]),np.zeros((3,3))),1),np.concatenate((np.zeros((3,3)),np.transpose(bodyToWorld[:3,:3])),1)),0)
            TF = np.zeros(FT.shape)
            TF[:3] = FT[3:]
            TF[3:] = FT[:3]
            bodyTF = np.dot(rotrot,TF)
        
            ξsc = np.concatenate([Ωsc,Vsc])
            ΩscX = buildSkew(Ωsc)
            VscX = buildSkew(Vsc)
            ξscV = np.concatenate((np.concatenate((ΩscX,Vsc),1),np.zeros((1,4))),0)
            
            adξ = np.concatenate((np.concatenate((ΩscX,np.zeros((3,3))),1),np.concatenate((VscX,ΩscX),1)),0)
            adξs = np.transpose(adξ)
            
            
            rightSide = np.dot(adξs,np.dot(self.Isc,ξsc)) +  bodyTF
            
            ξsc_dot = np.dot(np.linalg.inv(self.Isc), rightSide)
            bodyToStrange = np.eye(4)
            bodyToStrange[:3,:3] = bodyToWorld[:3,:3]
            bodyToWorld_dot = np.dot(bodyToStrange,ξscV) #gsc_dot = gsc * ξscV
            
            bodyToWorld+= bodyToWorld_dot*self.dt
            bodyToWorld = repairRotMatrix(bodyToWorld)

            
            Ωsc += ξsc_dot[:3,:] * self.dt
            Vsc += ξsc_dot[3:,:] * self.dt
        self.time += time.time() - start
        return (bodyToWorld,Ωsc,Vsc)
   # def setState(self,state):
   #     self.state = np.copy(state)
def sphereProperties(density,radius,library=np):
    volume = 4.0/3 * np.pi*(radius**3)
    mass = volume * density
    I = library.eye(3)*(2.0/5) *mass * (radius**2)
    return mass, I
def boatProperties(cylinder_mass,radius,cylinder_height,base_mass):
    base_x = (1/4) * base_mass *radius*radius
    base_y = base_x
    base_z = (1/2)*base_mass * radius * radius
    
    cylinder_x = (cylinder_mass/12) * (3*(radius**2+radius**2) + cylinder_height**2)
    cylinder_y = cylinder_x
    cylinder_z = (cylinder_mass/2) * (radius**2+radius**2)
    
    bases = 2
    cylinders = 1.5
    mass = base_mass * bases + cylinders* cylinder_mass
    MOI = (base_x*bases + cylinder_x*cylinders,
           base_y*bases + cylinder_y*cylinders,
           base_z*bases + cylinder_z*cylinders)
    MOI = np.diag(MOI)
    return mass,MOI
    
    