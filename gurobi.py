# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:10:00 2019

@author: tabor
"""

from gurobipy import *
import numpy as np
import time

#matrix multiply
class gurobiSolver:
    def __init__(self,dims,allowedDimsPerStep = None):
        self.dims = dims
        if allowedDimsPerStep == None:
            self.allowedDims = self.dims
        else:
            self.allowedDims = allowedDimsPerStep
    def absSolverSetup(self,A,B,m):
            u = m.addVars(self.dims,1)#,lb = [-10]*self.dims,ub = [10]*self.dims)
            z = m.addVars(self.dims,1)
            m.update()
            u = np.array([[u[i,0]] for i in range(self.dims)])
            z = np.array([[z[i,0]] for i in range(self.dims)])

            
            #print(A.shape)
            #print(u.shape)
            au = np.dot(A,u) #MM(A,u,None,None,1)
            bz = np.dot(B,z) #MM(B,z,None,None,1)


            for i in range(self.dims):
                u[i,0].lb = -1000.0
                u[i,0].ub = 1000.0
                z[i,0].ub = 1000.0
            m.update()
            if(controlOnOff is None):
                controlOnOff = m.addVars(self.dims,1, vtype=GRB.BINARY)
                controlOnOff = np.array([[controlOnOff[i,0]] for i in range(self.dims)])
            u2 = controlOnOff*u#u2 = vectorElementMult(controlOnOff,u,self.dims)
            m.update()

            newY = au + bz#MA(au,bz,6)

            for i in range(self.dims):
                m.addConstr(z[i,0]==abs_(u[i,0]))
            self.equalityConstraint(m,u,u2)  
            #for i in range(self.dims):
            #    m.addConstr(u2[i,0]==u[i,0])
                
            numMagnetsOn = np.sum(controlOnOff) #VD(np.ones([self.dims,1]),controlOnOff,self.dims)
            m.addConstr(numMagnetsOn==self.allowedDims)
            return newY,u,z,controlOnOff  
    def dummySolverSetup(self,A,B,m):
        u = m.addVars(6,1)#,lb = [-10]*self.dims,ub = [10]*self.dims)
        m.update()
        u = np.array([[u[i,0]] for i in range(6)])
        for i in range(6):
            u[i,0].lb = -0.1
            u[i,0].ub = 0.1
        m.update()   
        return u,u,u,u        
    def prune(self,array):
        for i in range(array.shape[0]):
            j=0
            size = array[i,0].size()
            while j < size:
                if(array[i,0].getCoeff(j) ==0):
                    array[i,0].remove(j)
                    size = array[i,0].size()
                else:
                    j = j+1
    def equalityConstraint(self,m,a,b):
        scale = 1#0**8

        for i in range(a.shape[0]):
            m.addConstr(scale*a[i,0]==scale*b[i,0])
    def constructNumpyVar(self,m,dims):
        var = m.addVars(dims,1)
        m.update()
        npVar = np.array([[var[i,0]] for i in range(dims)])
        for i in range(self.dims):
            npVar[i,0].lb = -100.0
            npVar[i,0].ub = 100.0
        m.update()
        return npVar
    def solvePlan(self,steps,model,X,simulator):
            
        scale = 10**4

        m = Model("mip")
        m.Params.NumericFocus = 3
        #m.Params.PSDTol = 10**6
        m.Params.TimeLimit = 60 * 60 * 5
        m.Params.FeasibilityTol = 10**-3
        state = simulator.state
        print('starting')
        A,B = model.getAB(state)
        
        simStartState = self.constructNumpyVar(m,12)
        
        info = []
        states = []
        states.append(simStartState)

        controls = []
        try:
            self.equalityConstraint(m,simStartState,state)
            state = simStartState
            for i in range(steps):
                
                newY,u,z,controlOnOff = self.absSolverSetup(A,B,m)
                self.prune(newY)

                controls.append([u[j,0] for j in range(self.dims)])
                
                nextState = simulator.forwardPass(state,newY)
                
                info.append((newY,u,controlOnOff,nextState,z))

                self.prune(nextState)

                state = self.constructNumpyVar(m,12)
                states.append(state)

                self.equalityConstraint(m,nextState,state)

                print('setup ' + str(i))
            m.update()
            controls = np.array(controls)

            #self.equalityConstraint(m,X,state)
            l2Control = np.sum(np.sum(np.multiply(controls,controls)))
            error = (state - X) *scale
            ob = np.dot(np.transpose(error),error) + l2Control

            
            m.setObjective(ob[0,0], GRB.MINIMIZE)
            m.optimize()
        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))
            
        np_states = np.zeros((len(states),states[0].shape[0],states[0].shape[1]))
        for j in range(np_states.shape[0]):
            for k in range(np_states.shape[1]):
                for l in range(np_states.shape[2]):
                    np_states[j][k][l] = states[j][k][l].x
        return m,info,np_states
    def solveWrench(self,A,B,Y,param = 0):
        try:
            m = Model("mip")
            scale = 10**4
            m.Params.OutputFlag = 0        
            
            newY,u,z,controlOnOff = self.absSolverSetup(A,B,m)
 
            error = (newY - Y)*scale #MA(newY,Y*-1,6) * scale
            #print(error)
            ob = np.dot(np.transpose(error),error) #VD(error,error,6)
            #print(ob.shape)
            m.setObjective(ob[0,0], GRB.MINIMIZE)

            m.optimize()
            best = np.array([v.x for v in m.getVars()])
            unp = np.reshape(best[:self.dims],(-1,1))
            znp = np.reshape(best[self.dims:self.dims*2],(-1,1))

            newY = np.matmul(A,unp) + np.matmul(B,znp) 
            #print('Y ',newY)
            
#            for v in m.getVars():
#                print('%s %g' % (v.varName, v.u))
        
            #print('Obj: %g' % m.objVal)
            
            bestU = np.reshape(best[:self.dims],(-1,1))
            bestControlChoice = np.rint(best[self.dims*2:],)
            chosenU = np.reshape(bestControlChoice,(-1,1))*bestU
            

            return chosenU
        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))
        
#        except AttributeError:
#            print('Encountered an attribute error')
        
