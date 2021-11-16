#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 12:31:38 2021

@author: tabor
"""
from models import multiMagModel
from modelOptimization import multiMagSolver
from environments import PlanarProblem,SpaceProblem

doBrute = False

if __name__ == '__main__':
    model = 5
    sphereCosts = []
    cylinderCosts = []
    bruteSphereCosts = []
    sphereModel = multiMagModel(env.magnets,diameter,True)
    sphereSolver = multiMagSolver(sphereModel,env.trajectoryDimensions,True,10)
    
    cylinderModel = multiMagModel(env.magnets,diameter,False)
    cylinderSolver = multiMagSolver(cylinderModel,env.trajectoryDimensions,False,10)
    for desired_wrench in desiredWrenchs:

        
        
        control1 =  sphereSolver.solveWrench(state[0][:3,3],desired_wrench)
        trueWrench1 = sphereModel.forwardPass(state[0][:3,3],control1)
        wrenchError1 = trueWrench1-desired_wrench
        cost2 = np.sum((wrenchError1[:3,:]**2)) + 200*np.sum((wrenchError1[3:,:]**2))
        sphereCosts.append(cost2)
        
        control2 =  cylinderSolver.solveWrench(state[0][:3,3],desired_wrench)
        trueWrench2 = cylinderModel.forwardPass(state[0][:3,3],control2)
        wrenchError2 = trueWrench2-desired_wrench
        cost2 = np.sum((wrenchError2[:3,:]**2)) + 200*np.sum((wrenchError2[3:,:]**2))
        cylinderCosts.append(cost2)
        
        if doBrute:
            control3 =  sphereSolver.bruteSphereSolver(state[0][:3,3],desired_wrench)
            trueWrench3 = sphereModel.forwardPass(state[0][:3,3],control3)
            wrenchError3 = trueWrench3-desired_wrench
            cost3 = np.sum((wrenchError3[:3,:]**2)) + 200*np.sum((wrenchError3[3:,:]**2))
            bruteSphereCosts.append(cost3)
        else:
            bruteSphereCosts.append(0.0)

        
    plt.figure()    
    plt.plot(np.array(sphereCosts)-np.array(bruteSphereCosts),'r')
    plt.plot(np.array(cylinderCosts)-np.array(bruteSphereCosts),'b')
    
    plt.figure()    
    plt.plot(np.array(sphereCosts)/np.array(cylinderCosts))