# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 19:35:46 2022

@author: tabor
"""
import numpy as np
import time
from models import SphereControlCommand,deconstruct_omega,modelCallRotatedWrench
import ForceFinder
import math



class devinSolver2:
    def __init__(self,model):
        self.model = model
        self.count = 0
        self.time = 0
        
        self.max_frequency = 15
        self.max_dipole_strength = 40
        
        self.mu0 = 4*np.pi*10**-7
        
        self.duty_time = 4
        min_full_rotations = 3
        self.duty_percent = min_full_rotations/ ( self.max_frequency * self.duty_time)
        

    def calculate_duty_cycling_and_angles(self,position,wrench):

        # Grab the desired Force
        Fdes = np.copy(wrench[:3,0])
        
        # Get the positions from magnets to object
        p1,p2 = -1*(np.array(self.model.magnets) - np.transpose(position))
        if Fdes[0] < 0:
            mag1 = 0
            mag2 = 0
        else:
            mag1 = 1
            mag2 = 1
        
        # Define FEA or Exp coefficients (1 = FEA, O = Exp) and initial search
        # parameters and precisions
        material = 'copper'
        lambda1 = np.linalg.norm(p1)
        mmax = self.max_dipole_strength
        r = self.object_properties.r        
        FEA_Exp = 1
        dtheta = 0.1
        dtheta_precision = 1e-8
        ddelta = 0.1
        ddelta_precision = 1e-8
        
        # Determine the parrallel and perpendicular forces and the rotation about
        # the x-axis
        fxd = abs(Fdes[0])
        fyzd = math.sqrt(Fdes[1]**2 +  Fdes[2]**2)



        # Handle the special case where thes desired force is entirely in the
        # x-direction
        ix = np.array([1, 0, 0])
        fdes_perp = np.array([0,Fdes[1],Fdes[2]])
        if np.linalg.norm(fdes_perp) == 0:
            w1hat = np.array([1, 0, 0])
            w1 = self.max_frequency*w1hat
            if Fdes[0] < 0:
                F, Fratio, T = ForceFinder(w1,mmax,material,FEA_Exp,r,lambda1)
            else:
                F, Fratio, T = ForceFinder(w1,mmax,material,FEA_Exp,r,-lambda1)
            if np.linalg.norm(Fdes) >= np.linalg.norm(F):
                return F, T, w1, -w1, 0, 0, mag1, mag2, 0, 1, mmax, mmax, 0, 0, 0.5, 0.5, 0, 0
            else:
                Fmax_ratio = np.linalg.norm(Fdes)/np.linalg.norm(F);
                mratio = math.sqrt(Fmax_ratio);
                m1_set = mratio*mmax
                if Fdes[0] < 0:
                    F, Fratio, T = ForceFinder(w1,m1_set,material,FEA_Exp,r,lambda1)
                else:
                    F, Fratio, T = ForceFinder(w1,m1_set,material,FEA_Exp,r,-lambda1)
                return F, T, w1, -w1, 0, 0, mag1, mag2, 0, 1, m1_set, m1_set, 0, 0, 0.5, 0.5, 0, 0
        
        # Now handle all other cases
        else:
            fhat_des_perp = fdes_perp/np.linalg.norm(fdes_perp)
            if Fdes[0] >= 0:
                w1hat = np.cross(ix,fhat_des_perp)
            elif Fdes[0] < 0:
                w1hat = np.cross(fhat_des_perp,ix)   
            w1 = wmag*w1hat
            
            if Fdes[0] < 0:
                F, Fratio, T = ForceFinder(w1,mmax,material,FEA_Exp,r,lambda1)
            else:
                F, Fratio, T = ForceFinder(w1,mmax,material,FEA_Exp,r,-lambda1)
            
            # Determine if desired force can be created using a single rotating dipole
            Fdes_ratio = np.linalg.norm(fdes_perp)/abs(Fdes[0])
            if Fdes_ratio <= Fratio:
                theta = math.pi/2
                
                while dtheta > dtheta_precision:
                    
                    Ry = np.array([[math.cos(math.pi/2 - theta),  0, math.sin(math.pi/2 - theta)],
                                  [         0               ,  1,           0             ],
                                  [-math.sin(math.pi/2 - theta), 0, math.cos(math.pi/2 - theta)]])

                    w1hatb = np.matmul(Ry,w1hat)
                    w1b = wmag*w1hatb
                    w1b2 = np.array([-1, 1, 1])*w1b

                    if Fdes[0] < 0:
                        F1, Fratio1, T1 = ForceFinder(w1b,mmax,material,FEA_Exp,r,lambda1)
                        F2, Fratio2, T2 = ForceFinder(w1b2,mmax,material,FEA_Exp,r,lambda1)
                    else:
                        F1, Fratio1, T1 = ForceFinder(w1b,mmax,material,FEA_Exp,r,-lambda1)
                        F2, Fratio2, T2 = ForceFinder(w1b2,mmax,material,FEA_Exp,r,-lambda1)
                    F  = (F1 + F2)/2
                    T = (T1 + T2)/2
                    Fratio = (Fratio1 + Fratio2)/2
                    
                    if Fdes_ratio == Fratio:
                        dtheta = dtheta_precision
                    elif Fratio > Fdes_ratio:
                        theta = theta - dtheta;
                    else:
                        dtheta = dtheta/2;
                        theta = theta + dtheta;
                        
                
                if np.linalg.norm(Fdes) >= np.linalg.norm(F):
                    return F, T, w1b, w1b2, 0, 0, mag1, mag2, 0, 1, mmax, mmax, 0, 0, 0.5, 0.5, 0, 0
                else:
                    Fmax_ratio = np.linalg.norm(Fdes)/np.linalg.norm(F);
                    mratio = math.sqrt(Fmax_ratio);
                    m1_set = mratio*mmax
                    if Fdes[0] < 0:
                        F1, Fratio1, T2 = ForceFinder(w1b,m1_set,material,FEA_Exp,r,lambda1)
                        F2, Fratio2, T2 = ForceFinder(w1b2,m1_set,material,FEA_Exp,r,lambda1)
                    else:
                        F1, Fratio1, T2 = ForceFinder(w1b,m1_set,material,FEA_Exp,r,-lambda1)
                        F2, Fratio2, T2 = ForceFinder(w1b2,m1_set,material,FEA_Exp,r,-lambda1)
                    return (F1+F2)/2, (T1+T2)/2, w1b, w1b2, 0, 0, mag1, mag2, 0, 1, m1set, m1set, 0, 0, 0.5, 0.5, 0, 0
            else:
                d1 = 0.5
                d2 = 0.5
                
                while ddelta > ddelta_precision:
                    F1, Fratio1, T1 = ForceFinder(w1b,mmax,material,FEA_Exp,r,-lambda1)
                    F2, Fratio2, T2 = ForceFinder(w1b,mmax,material,FEA_Exp,r,lambda1)
                    F = F1 + F2;
                    Fratio = np.linalg.norm(np.array([0,F[1],F[2]]))/abs(F[0])
                    
                    if Fdes_ratio = Fratio:
                        ddelta = ddelta_precision
                    elif Fratio > Fdes_ratio:
                        if Fdes[0] < 0:
        
      

        
    def solveWrench(self,position,desired_wrench):
        start = time.time()
        model_id =   self.count % (2 * len(self.model.magnets))    
        self.count += 1
        if model_id == 0:
            F,T,w1, w2, w3, w4, mag1, mag2, mag3, mag4, m1_set, m2_set, m3_set, m4_set, d1, d2, d3, d4 = self.calculate_duty_cycling_and_angles(position, desired_wrench)
            #print(m1_set, m2_set, m3_set, m3_set, d_yz, d_x)
            mags = [mag1,mag2,mag3,mag4]
            
            #rescale everything so a magnet is on for increments of self.duty_percent of the time
            scale_factor_t1 = 1 
            scale_factor_t2 = 1 
            
            if np.isnan(d1) or np.isnan(d2) or d_x==0.0 or d_yz==0.0:
                d_x = 0 if np.isnan(d_x) else d_x
                d_yz = 0 if np.isnan(d_yz) else d_yz
                self.T1 = d_yz
                self.T2 = d_x
            else:
                #rescale smaller one down, then adjust accordingly
                if(d_yz < d_x):
                    self.T1 = self.duty_percent*int(d_yz/self.duty_percent)
                    self.T2 = 1/2 - self.T1
                    if self.T2<0.5:
                        scale_factor_t2 = np.sqrt((d_x*self.T1)/(d_yz*self.T2))
                else:
                    self.T2 = self.duty_percent*int(d_x/self.duty_percent)
                    self.T1 = 1/2 - self.T2
                    if self.T1<0.5:
                        scale_factor_t1 =  np.sqrt((d_yz*self.T2)/(d_x*self.T1))
                
            
            dipole_1 = m1_set * scale_factor_t1
            dipole_2 = m2_set * scale_factor_t1
            dipole_3 = m3_set * scale_factor_t2
            dipole_4 = m3_set * scale_factor_t2
            
            self.dipoles = [dipole_1,dipole_2,dipole_3,dipole_4]
            self.omegas = [w1/np.linalg.norm(w1),
                           w2/np.linalg.norm(w2),
                           w3/np.linalg.norm(w3),
                           w3b/np.linalg.norm(w3b)]            
            self.controls = []
            
            actual_wrench = np.zeros((6,1))
            
            print('')
            print('')
            print('Times')
            print(self.T1)
            print(self.T2)
            print('')
            print('')
            
            
            
            for i in range(2 * len(self.model.magnets)):
                angles = deconstruct_omega(self.omegas[i])
                
                
                control = SphereControlCommand(mag_num = mags[i],
                                               dipole_strength=self.dipoles[i],
                                               psi=angles[0],xi=angles[1])
                if i <=1:
                    control.seconds = self.T1 * self.duty_time
                else:
                    control.seconds = self.T2 * self.duty_time
                
                actual_wrench += self.model.forwardSphere(position,control)*control.seconds
                #print(F*self.duty_time,T*self.duty_time)
                self.controls.append(control)
            print('')
            print('')
            print(actual_wrench/4)
            print(desired_wrench)
           
            print('')
            print('')

        control = self.controls[model_id]

        
        #print('chosen control',control)
        self.time += time.time()-start
        return control
