# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:41:12 2019

@author: tabor
"""
from oct2py import octave
import numpy as np
import time

class pythonModel:
    def __init__(self,magnets,fast=True):
        self.magnets = magnets
        self.dims = len(magnets)*3
        self.A = None
        self.B = None
        self.fast = fast
        self.time = 0.0
    def getAB(self,pose):
        if(self.fast and self.A is not None):
            return self.A,self.B
        position = pose[:3]
        newMagnets = (np.array(self.magnets) - np.transpose(position)).tolist()
        sigma_copper = 5.8E+07;
        sigma_al = 3.77E+07;
        sigma = sigma_copper;
        do = 0.05;
        start = time.time()
        allFT,A,B = self.calcEddyInput(newMagnets,sigma,do,15)
        runTime = time.time() - start
        self.time = self.time + runTime
        A=A[:,:self.dims] #* 10**6
        B=B[:,:self.dims] #* 10**6
        self.A = A
        self.B = B
        return A,B
    def forwardPass(self,pose,control):
        A,B = self.getAB(pose)
        #print(A.shape)
        #print(B.shape)
        #print(control.shape)
        FT = np.add(np.dot(A,control),np.dot(B,np.abs(control)))
        return FT
    def calcEddyInput(self, omnimagnets, sigma, DO, freq):
    # Matrix Pre-allocation
        AllFT_A = np.zeros((6, 3*len(omnimagnets))); 
        AllFT_B = np.copy(AllFT_A); 
        # Calculate the Modeled Input Matrix from Omnimagnet Locations 
        for omni in range(len(omnimagnets)):
            #print(omni)
            pos = omnimagnets[omni]
            omni_A, omni_B, omniFrame = self.omniAB_matrix(pos,sigma,DO,freq)
            #print(omni_A.shape)
            # Rotate Omnimagnet FT to the World Frame of the tool 
            omniworld_A  = np.concatenate([np.dot(omniFrame,omni_A[:3,:]), np.dot(omniFrame,omni_A[3:6,:])])
            omniworld_B  = np.concatenate([np.dot(omniFrame,omni_B[:3,:]), np.dot(omniFrame,omni_B[3:6,:])])
            # Keeps tracks the different rotations to each omnimagnet
            #print(omniworld_A.shape)
            #print(AllFT_A[:,(3*(omni)):(3*(omni+1))].shape)
            AllFT_A[:,(3*(omni)):(3*(omni+1))] = omniworld_A
            AllFT_B[:,(3*(omni)):(3*(omni+1))] = omniworld_B 
        AllFT = None
        return [AllFT, AllFT_A, AllFT_B]
    def omniAB_matrix(self,omni_pos, sigma, Do, freq):
        pi = 3.1416
        
        mu = 4*pi*np.power(10.0,-7);           #Magnetic Permittivity of Free Space
        
        # Eddy Current Ansys Modeling 
        model_C = np.array([[ 1, -4.39, -0.230, 0.00143, -7.38 ],      # Xaxis: Fx
                    [0, 0, 0, 0, 0]      ,                     # Xaxis: Fy = 0
                    [1, -3.27, -0.141, -0.000796, -7.26],  # Xaxis: Fz
                    [0, 0, 0, 0, 0],                           # Xaxis: Tx = 0
                    [1, -3.59, -0.127, -0.00150, -6.27],     # Xaxis: Ty
                    [0,0, 0, 0, 0],                           # Xaxis: Tz
                    [0, 0, 0, 0, 0],                           # Yaxis: Fx = 0
                    [1, -4.97, -0.193, 0.000377, -6.81 ],  # Yaxis: Fy 
                    [0, 0, 0, 0, 0],                          # Yaxis: Fz
                    [0, 0, 0, 0, 0],                           # Yaxis: Tx = 0
                    [1, -4.13, -0.101, -0.000957, -5.64],   # Yaxis: Ty
                    [0, 0, 0, 0, 0]] )                            # Yaxis: Tz = 0
               
        # Organize for omnimagnet transformation 
        # Y-axis Modeling a (constant) and b (power) coefficients
        Yaxis_Fabs = model_C[6+1,1:]
        Yaxis_T = model_C[6+4,1:]
        
        # X-axis Modeling
        Xaxis_Fabs = model_C[0,1:]
        Xaxis_Flin = model_C[2,1:]
        Xaxis_T = model_C[4,1:]
        
        # Build Matrices for Linear (A) components and Non-linear (B) components
        
        # Linear:
        C_A = np.zeros((6,3,5));
        C_A[:,:,4] = np.zeros((6,3));
        #X axis                                         Y axis                                          Z axis
        0;                                              0;                                              0 ;                         # Fx
        0;                                              0;                                              C_A[1,2,4] = 1; C_A[1,2,:4] = Xaxis_Flin;          # Fy
        0;                                              C_A[2,1,4]= -1 ; C_A[2,1,:4] = Xaxis_Flin;     0;                          # Fz
        C_A[3,0,4] = 1; C_A[3,0,:4] = Yaxis_T;          0;                                              0;                                   # Tx
        0;                                              C_A[4,1,4]= -1 ; C_A[4,1,:4] = Xaxis_T;        0;                          # Ty
        0;                                              0;                                              C_A[5,2,4] = -1; C_A[5,2,:4] = Xaxis_T;           # Tz
        
        # Absolute Terms
        C_B = np.zeros((6,3,5));
        C_B[:,:,4] = np.zeros((6,3));
        C_B[0,0,:4] = Yaxis_Fabs; C_B[0,1,:4] = Xaxis_Fabs; C_B[0,2,:4] = Xaxis_Fabs;
        C_B[0,0,4] = 1; C_B[0,1,4] = 1; C_B[0,2,4] = 1;
        
        omni_A = np.zeros((6,3))
        omni_B = np.zeros((6,3))
        
        # Calculate the Modeled Input Matrix from Omnimagnet Locations
        pos = omni_pos
        dist = np.sqrt(np.dot(pos,pos))
        D = dist;
        #
        P1 = sigma*mu*freq*Do*Do;
        P2 = D/Do;
        
        # Build master force-torque matrix dependent on location
        for i in range(6):
            for j in range(3):
                if i < 3:
                    # Force
                    omni_A[i,j] = C_A[i,j,4]* ( 10**( C_A[i,j,0]*P1**C_A[i,j,1] ) * P2**( C_A[i,j,2]*P1+C_A[i,j,3]) ) * mu/(Do**4);
                    omni_B[i,j] = C_B[i,j,4]* ( 10**( C_B[i,j,0]*P1**C_B[i,j,1] ) * P2**( C_B[i,j,2]*P1+C_B[i,j,3]) ) * mu/(Do**4);
                else:
                    pass
                    #Torque
                    omni_A[i,j] = C_A[i,j,4]* ( 10**( C_A[i,j,0]*P1**C_A[i,j,1] )*P2**(C_A[i,j,2]*P1+C_A[i,j,3]) ) * mu/(Do**3);
                    omni_B[i,j] = C_B[i,j,4]* ( 10**( C_B[i,j,0]*P1**C_B[i,j,1] )*P2**(C_B[i,j,2]*P1+C_B[i,j,3]) ) * mu/(Do**3);
        
        
        omni_FT = omni_A + omni_B;
        
        # ----- Construct Rotation Vector from Omni Frame to World Frame ---%
        # X vector points toward tool
        omni_X = -1*(pos/dist);
        
        # Set default Z vector as upward
        Z = np.array([0, 0, 1])
        #
        # Determine if default Z and X are the same
        if np.all(abs(omni_X) == abs(Z)):
            #         fprintf('X and Z are the same vectors, new perpendicular Z chosen [0 1 0]\n');
            Z = np.array([0, 1, 0])#set Z to be something else if already equal
        
        
        # Determine Z component perpendicular to Y axis
        if np.dot(omni_X,Z) == 0:
            #         fprintf('Z and X are already perpendicular\n');
            omni_Z = Z;
        else:
            omni_Z = (omni_X)*np.sin( np.arccos(np.dot(omni_X,Z)));
        
        
        # Obtain Y axis by crossing Z and X
        omni_Y = np.cross(omni_Z,omni_X);
        
        # Combine to build rotation matrix
        omniFrame = np.transpose(np.array([omni_X, omni_Y, omni_Z]));
        return (omni_A,omni_B,omniFrame)        
