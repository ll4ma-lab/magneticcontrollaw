# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:10:00 2019

@author: tabor
"""

from gurobipy import *
import numpy as np
import time
from pythonModel import pythonModel

#matrix multiply
class gurobiSolver:
    def __init__(self,dims,allowedDimsPerStep = None):
        self.dims = dims
        if allowedDimsPerStep == None:
            self.allowedDims = self.dims
        else:
            self.allowedDims = allowedDimsPerStep
    def absSolverSetup(self,A,B,m,controlOnOff=None):
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
                u[i,0].lb = -1600.0
                u[i,0].ub = 1600.0
                z[i,0].ub = 1600.0
            m.update()
            if(controlOnOff is None):
                controlOnOff = m.addVars(self.dims,1, vtype=GRB.BINARY)
                controlOnOff = np.array([[controlOnOff[i,0]] for i in range(self.dims)])
            u2 = controlOnOff*u#u2 = vectorElementMult(controlOnOff,u,self.dims)
            m.update()

            newY = au + bz#MA(au,bz,6)

            for i in range(self.dims):
                m.addConstr(z[i,0]==abs_(u[i,0]))
            equalityConstraint(m,u,u2)  
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
    def buildQ(self,stateCostString,loops = 1):
        
        activated = [0]*(loops * 6)
        if("all" in stateCostString):
            activated = [1]*(loops * 6)
        letterOptions = ["x","y","z","r","p","t"]
        for j in range(loops):
            for i in range(len(letterOptions)):
                if(letterOptions[i] in stateCostString):
                    index = j*6+i
                    #print(index)
                    activated[index] = 1
        Q = np.diag(activated)
        return Q
        
    def solvePlan(self,steps,model,X,simulator,stateCostString="all",keepControlSteps=1):
        Q = self.buildQ(stateCostString,2)
        print(stateCostString)
        print(Q)
        scale = 1#10**4
        UnitConversionFactor = 10**6 # change length unit used internally. 1 being meters 10**3 being milimeters

        #scale units inside plan 
        simulator = simulator.getScaledSimulator(UnitConversionFactor)
        X = X * UnitConversionFactor
        
        
        m = Model("mip")
        model = symbolicModel(model,m)
        m.Params.NumericFocus = 3
        m.Params.NonConvex = 2
        #m.Params.PSDTol = 10**6
        m.Params.TimeLimit = 60 * 5
        #m.Params.FeasibilityTol = 10**-3
        state = simulator.state
        print('starting')
        A,B = model.getAB(state)
        simStartState = constructNumpyVar(m,12)
        
        info = []
        states = []
        states.append(simStartState)

        controls = []
        try:
            equalityConstraint(m,simStartState,state)
            state = simStartState
            for i in range(steps):
                if(i%keepControlSteps==0):
                    print('made new controlOnOff')
                    newY,u,z,controlOnOff = self.absSolverSetup(A,B,m)
                    print('nop')
                else:
                    print('used old controlOnOff')
                    newY,u,z,controlOnOff = self.absSolverSetup(A,B,m,controlOnOff)

                prune(newY)

                controls.append([u[j,0] for j in range(self.dims)])
                print('preFP')
                nextState = simulator.forwardPass(state,newY)
                
                info.append((newY,u,controlOnOff,nextState,z))
                
                prune(nextState)
                state = constructNumpyVar(m,12)
                states.append(state)
                print('pre constraint')
                equalityConstraint(m,nextState,state)
                print(state)
                A,B = model.getAB(state/UnitConversionFactor)
                print('setup ' + str(i))
            m.update()
            controls = np.array(controls)

            #equalityConstraint(m,X,state)
            l2Control = np.sum(np.sum(np.multiply(controls,controls)))
            error = (state - X) *scale
            print(error)
            ob = np.dot(np.dot(np.transpose(error),Q),error) + l2Control * 100
            #print(ob)
            prune(ob)
            print(ob)
            
            m.setObjective(ob[0,0], GRB.MINIMIZE)
            m.optimize()
        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))
            
        np_states = np.zeros((states[0].shape[0],len(states)))
        for j in range(np_states.shape[0]):
            for k in range(np_states.shape[1]):
                np_states[j][k] = states[k][j][0].x / UnitConversionFactor
                
        controls = []
        for i in range(steps):
            control_sym = info[i][1]
            On_OFF_sym = info[i][2]
        
            control = np.zeros(control_sym.shape)
            for j in range(control_sym.shape[0]):
                control[j,0]=control_sym[j,0].x * On_OFF_sym[j,0].x
            controls.append(control)

        return m,info,np_states,controls
    def solveWrench(self,A,B,Y,stateCostString="all",param = 0):
        try:
            Q = self.buildQ(stateCostString)
            #print('loop')
            #print(Y)
            Y = np.dot(Q,Y)
            #print(Y)
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
            
            #print(chosenU.shape)
            return chosenU
        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))
            return np.reshape(np.array([0]*self.dims),(-1,1))

        except AttributeError:
            print('Encountered an attribute error')
            #chosenU = np.reshape(bestControlChoice,(-1,1))*bestU

            return np.reshape(np.array([0]*self.dims),(-1,1))
        
class symbolicModel(pythonModel):
    def __init__(self,dynamicsModel,gurobiModel):
        self.m = gurobiModel
        super(symbolicModel,self).__init__(dynamicsModel.magnets,dynamicsModel.fast)
    def getAB(self,pose):            
        A,B = super(symbolicModel,self).getAB(pose)
        if(self.fast):
            return A,B
        Anew = getEquivalent(self.m,A)
        Bnew = getEquivalent(self.m,B)
        return Anew,Bnew
    def omniAB_matrix(self,omni_pos, sigma, Do, freq):
        print('called')
        if(self.fast):
            return super(symbolicModel,self).omniAB_matrix(omni_pos, sigma, Do, freq)
        
        pos = np.reshape(np.array(omni_pos),(1,3))
        dist = arbitraryPower(self.m,np.reshape(np.dot(pos,np.transpose(pos)),(1,1)),np.array([[0.5]]))
        D = dist;
        print('part 1')
        
        P2 = D/Do;
        omni_A = np.zeros((6,3))
        omni_B = np.zeros((6,3))
        P2_array = np.ones((6,3)) * P2

        omni_A = np.multiply(self.AConstant,arbitraryPower(self.m,P2_array,self.APower))
        omni_B = np.multiply(self.BConstant,arbitraryPower(self.m,P2_array,self.BPower))
        print('part 2')

        
        # ----- Construct Rotation Vector from Omni Frame to World Frame ---%
        # X vector points toward tool
        #print(pos)
        overDist = arbitraryPower(self.m,dist,np.array([[-1]]))
        #print(overDist.shape)
        #print(pos.shape)       
        omni_X = -1*np.dot(overDist,pos)
        #print(omni_X)
        
        print('part 3')

        # Set default Z vector as upward
        Z = np.reshape(np.array([0, 0, 1]),(1,3))
        #
        # Determine if default Z and X are the same
        print('z ' + str(omni_X))
        omni_X = getEquivalent(self.m,omni_X)
        omni_Z = np.cross(omni_X,Z)
        overOmni_Z_length = arbitraryPower(self.m,np.reshape(np.dot(omni_Z,np.transpose(omni_Z)),(1,1)),np.array([[-0.5]]))
        omni_Z = np.dot(overOmni_Z_length,omni_Z)
        # Obtain Y axis by crossing Z and X
        print('part 4')

        omni_Z = getEquivalent(self.m,omni_Z)

        omni_Y = np.cross(omni_Z,omni_X);
                
        # Combine to build rotation matrix
        omniFrame = np.transpose(np.array([omni_X, omni_Y, omni_Z]))[:,0,:];
        omniFrame = getEquivalent(self.m,omniFrame)
        return (omni_A,omni_B,omniFrame)
def getEquivalent(m,npGurobiTerm):
    newTerm = constructNumpyVar(m,npGurobiTerm.shape[0],npGurobiTerm.shape[1])
    equalityConstraint(m,npGurobiTerm,newTerm)
    return newTerm
def arbitraryPower(m,base,power):
    #note The lower bound of variable x must be nonnegative
    #so whatever is the base will be limited to lb of 0
    newBase = constructNumpyVar(m,base.shape[0],base.shape[1])
    output = constructNumpyVar(m,base.shape[0],base.shape[1])
    equalityConstraint(m,newBase,base)
    print(power.shape)
    for j in range(base.shape[1]):
        for i in range(base.shape[0]):
            if(power[i,j]==0):
                 m.addConstr(output[i,j]==1)
            if(power[i,j]>0):
                newBase[i,j].lb = 0
                m.addGenConstrPow(newBase[i,j],output[i,j],float(power[i,j]))
            if(power[i,j]<0):
                output[i,j].lb = 0
                m.addGenConstrPow(output[i,j],newBase[i,j],float(-1*power[i,j]))
    return output
def prune(array):
    for i in range(array.shape[0]):
        j=0
        size = array[i,0].size()
        while j < size:
            if(array[i,0].getCoeff(j) ==0):
                array[i,0].remove(j)
                size = array[i,0].size()
            else:
                j = j+1
def equalityConstraint(m,a,b):
    scale = 1#0**8
    print('equality')    
    print(a)
    print(b)
    for j in range(a.shape[1]):
        for i in range(a.shape[0]):
            m.addConstr(scale*a[i,j]==scale*b[i,j])
def constructNumpyVar(m,dims,dims2=1,lim=1000000.0,):
    var = m.addVars(dims,dims2)
    m.update()
    npVar = np.array([[var[i,j] for j in range(dims2)] for i in range(dims)])
    for j in range(dims2):
        for i in range(dims):
            npVar[i,j].lb = -1*lim
            npVar[i,j].ub = lim
    m.update()
    return npVar
  
    