# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:41:12 2019

@author: tabor
"""
import numpy as np
import time
#from oct2py import octave
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf
class cylinderSolver():
    def __init__(self,magnets):
        self.magnets = magnets
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,use_per_session_threads=True))
        self.time = 0
    def buildQ(self,stateCostString,objectProperties,tolerances,loops=1):
        
        activated = [0]*(loops * 6)
        if("all" in stateCostString):
            activated = [1]*(loops * 6)
        letterOptions = ["x","y","z","r","p","t"]
        for j in range(loops):
            for i in range(len(letterOptions)):
                if(letterOptions[i] in stateCostString):
                    index = j*6+i
                    #print(index)
                    if i < 3:   
                        activated[index] = 1.0 #/ (objectProperties[0] * tolerances[0])
                    else: 
                        activated[index] = 380.0 #/ (objectProperties[1][0,0] * tolerances[1])

        Q = np.diag(activated)
        print(Q)
        return Q

    def setup(self,model,stateCostString,objectProperties,tolerances):
        s = time.time()
        powers = []
        thetas = []
        costs = []    
        ft = []
        train_ops= []
        Rs = []
        carts = []
        tf_vars = []
        one = tf.constant(1.0,dtype=tf.float64)
        zero = tf.constant(0.0,dtype=tf.float64)
        
        Y = tf.placeholder(tf.float64,shape = [6,1]) 
        magPoses = tf.placeholder(tf.float64,shape = np.array(self.magnets).shape) 
        Q = self.buildQ(stateCostString,objectProperties=objectProperties,tolerances=tolerances)

        self.Y = Y
        self.magPoses = magPoses
        self.allowedModels = 3
        for i in range(len(self.magnets)):
            
            chosenMagnetPose = magPoses[i]
            omniFrame = model.getOmniFrame(chosenMagnetPose,True)
            _,cart = model.modelCall(chosenMagnetPose,1,use_tf = True)
            omniFrame_t = tf.convert_to_tensor(omniFrame)
            cart = tf.stack([tf.stack(cart[i].tolist()) for i in range(6)])
            for j in range(self.allowedModels):

                powerPercent = tf.Variable(0,dtype=tf.float64)
                clippedPower = tf.sigmoid(powerPercent) * 40
                if j == 0:
                    anglePercent = tf.Variable(0.0,dtype=tf.float64)
                    clippedTheta = tf.sigmoid(anglePercent) * 720 - 360
                    tf_vars.append(anglePercent)
                    radians = clippedTheta*np.pi/180.0        
    
                    Rx=np.array([[one,zero,zero],
                              [zero,tf.cos(radians) , -tf.sin(radians)],
                              [zero,tf.sin(radians) , tf.cos(radians)]])
                    Rx_t = tf.stack([tf.stack(Rx[0].tolist()),tf.stack(Rx[1].tolist()),tf.stack(Rx[2].tolist())])
                    R = tf.matmul(omniFrame_t,Rx_t)
                else:
                    R = omniFrame_t

                Rs.append(R)
                cartT = cart * clippedPower * clippedPower
                carts.append(cartT)
                FT = tf.concat([tf.matmul(R,cartT[:3,j:j+1]),tf.matmul(R,cartT[3:,j:j+1])],0)  
                
                errors = FT-Y
                cost =1e10* tf.matmul(tf.matmul(tf.transpose(errors),Q),errors)
                
                
                if j ==0:
                    thetas.append(clippedTheta)
                ft.append(FT)
                powers.append(clippedPower)
                costs.append(cost)
                train_ops.append(tf.train.AdamOptimizer(0.5e-1).minimize(cost))
                tf_vars.append(powerPercent)

        print(3,time.time()-s)
        s = time.time()

        magNum = tf.argmin(tf.stack(costs))
        #cost = np.sum(costs)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print(4,time.time()-s)
        self.ops = [train_ops,costs,magNum,thetas,powers,ft,Rs,carts]
        self.tf_vars = tf_vars
    def solveWrench(self,model,state,wrench,other=False):
        s = time.time()
        position = state[0:3]
        
        myDict = {}
        myDict[self.Y] = wrench
        myDict[self.magPoses] = np.array(self.magnets) - np.transpose(position)
        #init_op = tf.global_variables_initializer()
        init_op = [self.tf_vars[i].initializer for i in range(len(self.tf_vars))]
        self.sess.run(init_op)

        _,bestCost,magChoice,bestThetas, ps, fts,rs,carts = self.sess.run(self.ops,feed_dict=myDict)

        #print('start Thetas',bestThetas)
        for i in range(50):
            _,bestCost,magChoice,bestThetas, ps, fts,rs,carts = self.sess.run(self.ops,feed_dict=myDict)
#            if(np.min(bestCost)>0.01):            
#                print(magChoice,[bestCost[j][0,0] for j in range(len(bestCost))])
        #print('wrench solve ',time.time()-s)
        mag = magChoice[0,0]
        magNum = int(mag/self.allowedModels)
        modelNumber = mag%self.allowedModels 
        theta = bestThetas[magNum]
        power = ps[mag]
        control = [magNum,modelNumber,power,theta]
        #print('')
        #model.forwardPass(state,control)
        #print(control)
        self.time +=  (time.time() - s)
        return control

        
    
    
class cylinderModel(object):
    def __init__(self,magnets,diameter):
        self.magnets = magnets
        self.dims = len(magnets)*3
        self.time = 0.0
        sigma_copper = 5.8E+07;
        sigma_al = 3.77E+07;
        self.sigma = sigma_copper;
        self.do = diameter;
        self.freq = 15
    def eddyFTRotated(self, dist, dipole_strength, freq, radii,use_tf=False):
        #Constants 
        sig = 5.8e7;          # conductivity of copper 
        mu = 4*np.pi*1e-7;        # Magnetic Permittivity of Free Space
        # --- Model Coefficients --- # 
        model_C = np.array([
            # --- Rho Configuration 
            # F Rho
            [1, 0.0718, 1.56,7.22],      
          # Xaxis: Fx
            
            [0,0,0, 0],
            [-1, 0.0858, 0.935,7.16],                # Xaxis: Fz
 
                       # Xaxis: Fy = 0
            # F Phi 
            [0,0,0, 0],                           # Xaxis: Tx = 0
            # T X
            [-1, 0.0408, 0.943, 6.17],                # Xaxis: Ty

            [0,0,0, 0],                           # Xaxis: Tz

            # --- +Z Configuration
            # F Z
            [1, 0.0146, 1.50,  6.82],              # Yaxis: Fy
            [0,0,0, 0],                           #  Yaxis: Fz
            [0,0,0, 0],                           # Yaxis: Fx = 0
        
            # T Z 
            [1, 0.000917, 0.956,  5.64],               # Yaxis: Ty
            [0,0,0,0,],
            [0,0,0, 0],                           # Yaxis: Tx = 0
        
             # --- -Z Configuration
            # F Z
            [1, 0.0146, 1.50,  6.82],              # Yaxis: Fy
            [0,0,0, 0],                           # Yaxis: Fz
            [0,0,0,0,],                           # Yaxis: Fx = 0
        
            # T Z 
            [-1, 0.000917, 0.956,  5.64],               # Yaxis: Ty
            [0,0,0,0],
            [0,0,0, 0]])                           # Yaxis: Tx = 0
        P1 = sig*mu*freq*radii*radii

        numerator = np.power(P1,model_C[:,2:3])
        
        P2 = dist/radii;
        denominator = np.power(P2,model_C[:,3:4] )


        P0_temp = model_C[:, 0:1]*model_C[:, 1:2]*numerator/denominator
        
        P0 = np.stack([P0_temp[:6] , P0_temp[6:12] , P0_temp[12:18] ],1)[:,:,0]

        FT_Cycle = np.copy(P0)
        c=  mu*np.power(dipole_strength,2)*np.power(radii,-4)
        FT_Cycle[:3,:] *= c
        FT_Cycle[3:,:] *= mu*np.power(dipole_strength,2)*np.power(radii,-3)
        
        FT_Cartesian = np.copy(FT_Cycle)
        #FT_Cartesian[2,0] *= -1
        return FT_Cycle,FT_Cartesian
        #model_C[:, 1] .*  .*P2.^( model_C[:,4]) 
        
        
        
        
        
        
        
        
        
        
    def modelCall(self,chosenMagnetPose,power,m=False,use_tf = False):
        if(m):
            cycle,cart = octave.eddyFTRotated(np.linalg.norm(chosenMagnetPose),power,self.freq,self.do/2.0,nout=2)
            return cycle,cart
        if(use_tf):
            cycle,cart = self.eddyFTRotated(tf.norm(chosenMagnetPose),power,self.freq,self.do/2.0,use_tf)
            return cycle,cart
        
        cycle,cart = self.eddyFTRotated(np.linalg.norm(chosenMagnetPose),power,self.freq,self.do/2.0)
        return cycle,cart
    def forwardPass(self,pose,control):
            #control[0] = magnetNumber
            #control[1] = modelNumber
            #control[2] = power
            #control[3] = theta
        s = time.time()
        position = pose[0:3]
        chosenMagnetPose = (np.array(self.magnets) - np.transpose(position)).tolist()[control[0]]
        omniFrame = self.getOmniFrame(chosenMagnetPose)
        _,cart = self.modelCall(chosenMagnetPose,control[2])
        if(control[1]==0):
            theta = control[3]
            thetarad = np.deg2rad(theta)
            Rx=np.array([[1,0,0],
                      [0,np.cos(thetarad) , -np.sin(thetarad)],
                      [0,np.sin(thetarad) , np.cos(thetarad)]])
            R = np.dot(omniFrame,Rx)
            FT = np.concatenate([np.dot(R,cart[:3,0]),np.dot(R,cart[3:,0])])
        if(control[1]==1):
            FT = np.concatenate([np.dot(omniFrame,cart[:3,1]),np.dot(omniFrame,cart[3:,1])])
        if(control[1]==2):
            FT = np.concatenate([np.dot(omniFrame,cart[:3,2]),np.dot(omniFrame,cart[3:,2])])
        self.time +=  (time.time() - s)
        return np.reshape(FT,(6,1))
    def getOmniFrame(self,pos,use_tf=False):
        Z = np.array([0, 0, 1])

        if(use_tf):
            dist = tf.norm(pos)
            omni_X = -1*(pos/dist);
            
    
            omni_Z = tf.cross(omni_X,Z)
            omni_Z = omni_Z/tf.norm(omni_Z)
                    
            
            # Obtain Y axis by crossing Z and X
            omni_Y = tf.cross(omni_Z,omni_X);
            # Combine to build rotation matrix
            omniFrame = tf.transpose(tf.stack([omni_X, omni_Y, omni_Z]));
        else:
            dist = np.linalg.norm(pos)
        
            omni_X = -1*(pos/dist);
            
    
            omni_Z = np.cross(omni_X,Z)
            omni_Z = omni_Z/np.sqrt(np.dot(omni_Z,omni_Z))
                    
            
            # Obtain Y axis by crossing Z and X
            omni_Y = np.cross(omni_Z,omni_X);
            # Combine to build rotation matrix
            omniFrame = np.transpose(np.array([omni_X, omni_Y, omni_Z]));
        return omniFrame
