# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:10:00 2019

@author: tabor
"""

from gurobipy import *
import numpy as np
#matrix multiply
def MM(a,b,AFirst = None,ASecond = None,BSecond = None,debug = False):
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
                if(a[i,k] == 0):
                    pass
                else:
                    s += a[i,k]*b[k,j]
                    if(debug):                    
                        print(a[i,k])
                        print(b[k,j])
                        print(a[i,k]*b[k,j])
            if(debug):
                print(s)
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
    first = True
    for i in range(0,length):
        if(first):
            workingSum = a[i,0] * b[i,0]
            first = False
        else:
            workingSum += a[i,0] * b[i,0]
    return workingSum

class gurobiSolver:
    def __init__(self,dims):
        pass
    def solve(self,A,B,Y,param = 0):
        dims = A.shape[1]
        try:
            m = Model("mip")
            scale = 10**4
            m.Params.OutputFlag = 0        
            x = m.addVars(dims,1)#,lb = [-10]*dims,ub = [10]*dims)
            u = m.addVars(dims,1)
            m.update()

            ax = MM(A,x,None,None,1)
            bu = MM(B,u,None,None,1)
            m.update()

            # Set objective
            newY = MA(ax,bu,6)
            #print(newY)
            error = MA(newY,Y*-1,6) * scale
            #print(error)
            #ob = error[1,0]*error[1,0] + error[0,0]*error[0,0] +  error[2,0]*error[2,0] + error[3,0]*error[3,0] + error[4,0]*error[4,0]
            ob = VD(error,error,6)
            #print(ob)
            m.setObjective(ob, GRB.MINIMIZE)
           # print(ax)

            for i in range(dims):
               # m.addConstr(u[i,0]<=x[i,0])
                #continue
                m.addConstr(u[i,0]==abs_(x[i,0]))
#            p = m.presolve()
#            p.printStats()
            #m.tune()
            #m.Params.Presolve = 0
            m.optimize()
            
            best = np.array([v.x for v in m.getVars()])
            xnp = np.reshape(best[:dims],(-1,1))
            unp = np.reshape(best[dims:],(-1,1))

            newY = np.matmul(A,xnp) + np.matmul(B,unp) 
            #print('Y ',newY)
            
#            for v in m.getVars():
#                print('%s %g' % (v.varName, v.x))
        
            #print('Obj: %g' % m.objVal)
        
        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))
        
#        except AttributeError:
#            print('Encountered an attribute error')
        return np.reshape(best[:dims],(-1,1))
