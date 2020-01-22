# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:10:00 2019

@author: tabor
"""

from gurobipy import *
import numpy as np
#matrix multiply
class gurobiSolver:
    def __init__(self,dims,allowedDimsPerStep = None):
        if allowedDimsPerStep == None:
            self.allowedDims = dims
        else:
            self.allowedDims = allowedDimsPerStep
    def solve(self,A,B,Y,param = 0):
        dims = A.shape[1]
        try:
            m = Model("mip")
            scale = 10**4
            m.Params.OutputFlag = 0        
            u = m.addVars(dims,1)#,lb = [-10]*dims,ub = [10]*dims)
            z = m.addVars(dims,1)
            m.update()
            u = np.array([[u[i,0]] for i in range(dims)])
            z = np.array([[z[i,0]] for i in range(dims)])

            
            #print(A.shape)
            #print(u.shape)
            au = np.dot(A,u) #MM(A,u,None,None,1)
            bz = np.dot(B,z) #MM(B,z,None,None,1)

            for i in range(dims):
                m.getVars()[i].lb = -100.0
                m.getVars()[i].ub = 100.0


            controlOnOff = m.addVars(dims,1, vtype=GRB.BINARY)
            controlOnOff = np.array([[controlOnOff[i,0]] for i in range(dims)])
            u2 = controlOnOff*u#u2 = vectorElementMult(controlOnOff,u,dims)
            m.update()


            # Set objective
            newY = au + bz#MA(au,bz,6)
            #print(newY)
            error = (newY - Y)*scale #MA(newY,Y*-1,6) * scale
            #print(error)
            ob = np.dot(np.transpose(error),error) #VD(error,error,6)
            #print(ob.shape)
            m.setObjective(ob[0,0], GRB.MINIMIZE)

            for i in range(dims):
                m.addConstr(z[i,0]==abs_(u[i,0]))
                
            for i in range(dims):
                m.addConstr(u2[i,0]==u[i,0])
                
            numMagnetsOn = np.sum(controlOnOff) #VD(np.ones([dims,1]),controlOnOff,dims)
            m.addConstr(numMagnetsOn==self.allowedDims)
#            p = m.presolve()
#            p.printStats()
            #m.Params.Presolve = 0

            m.optimize()
            #print(m.getVars())
            best = np.array([v.x for v in m.getVars()])
            unp = np.reshape(best[:dims],(-1,1))
            znp = np.reshape(best[dims:dims*2],(-1,1))

            newY = np.matmul(A,unp) + np.matmul(B,znp) 
            #print('Y ',newY)
            
#            for v in m.getVars():
#                print('%s %g' % (v.varName, v.u))
        
            #print('Obj: %g' % m.objVal)
            
            bestU = np.reshape(best[:dims],(-1,1))
            bestControlChoice = np.rint(best[dims*2:],)
            chosenU = np.reshape(bestControlChoice,(-1,1))*bestU
            

            return chosenU
        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))
        
#        except AttributeError:
#            print('Encountered an attribute error')
        
