# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:10:00 2019

@author: tabor
"""

from gurobipy import *
import numpy as np
#matrix multiply
def MM(a,b,AFirst = None,ASecond = None,BSecond = None):
    if(AFirst is None):
        AFirst = len(a)
    if(ASecond is None):
        ASecond = len(a[0])
    if(BSecond is None):
        BSecond = len(b[0])
    c = []
    for i in range(0,AFirst):
        temp=[]
        for j in range(0,BSecond):
            s = 0
            for k in range(0,ASecond):
                s += a[i,k]*b[k,j]
            temp.append(s)
        c.append(temp)
    return np.array(c)
#matrix add
def MA(a,b,length = None):
    if(length == None):
        length = len(a)
    c = []
    for i in range(0,length):
        c.append(a[i]+b[i])
    return np.array(c)
#vector dot
def VD(a,b,length = None):
    if(length ==None):
        length = len(a)
    workingSum = 0
    for i in range(0,length):
        workingSum += a[i,0] * b[i,0]
    return workingSum

class gurobiSolver:
    def __init__(self,dims):
        pass
    def solve(self,A,B,Y):
        dims = A.shape[1]
        try:
            m = Model("mip")
            m.Params.OutputFlag = 0        
            x = m.addVars(dims,1)#,lb = [-10]*dims,ub = [10]*dims)
            u = m.addVars(dims,1)
        
            ax = MM(A,x,None,None,1)
            bu = MM(B,u,None,None,1)
            # Set objective
            newY = MA(ax,bu,dims)
            error = MA(newY,Y*-1,dims)
            ob = error[0,0]*error[0,0]
            ob = VD(error,error,dims)
            m.setObjective(ob, GRB.MINIMIZE)
        
            # Add constraint: x + 2 y + 3 z <= 4
        
            # Add constraint: x + y >= 1
            for i in range(dims):
               # m.addConstr(u[i,0]<=x[i,0])
                m.addConstr(u[i,0]==abs_(x[i,0]))
            m.optimize()
            
            best = [v.x for v in m.getVars()]
            return best[:dims]
        
#            for v in m.getVars():
#                print('%s %g' % (v.varName, v.x))
#        
#            print('Obj: %g' % m.objVal)
        
        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))
        
        except AttributeError:
            print('Encountered an attribute error')