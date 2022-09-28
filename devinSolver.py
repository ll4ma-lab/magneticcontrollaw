# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 19:35:46 2022

@author: tabor
"""
import numpy as np
import time
from models import SphereControlCommand,deconstruct_omega,modelCallRotatedWrench



class devinSolver:
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
        

    def calculate_duty_cycling(self,position,wrench):
        # Determine the position vectors/unit vectors/normal unit vectors    
        p1,p2 = -1*(np.array(self.model.magnets) - np.transpose(position))
        print(p1,p2)
        Fdes = np.copy(wrench[:3,0])
        
        #seemed like different coordinate system 
        Fdes[1]*=-1
        
        
        if Fdes[0] < 0:
            p3 = p2;
        else:
            p3 = p1;

        # Determine the parrallel and perpendicular forces and the rotation about
        # the x-axis
        fxd = abs(Fdes[0])
        fyzd = np.sqrt(Fdes[1]**2 +  Fdes[2]**2)
        
        # Determine the direction for omega 1 and 2 based on fyzd
        ix = np.array([1, 0, 0])
        fdes_parallel = np.array([fxd, 0 ,0])
        fdes_perp = np.array([0,Fdes[1],Fdes[2]])
        fhat_des_perp = fdes_perp/np.linalg.norm(fdes_perp)
        
        w1hat = np.cross(ix,fhat_des_perp)
        w2hat = np.cross(fhat_des_perp,ix)
        w3hat = np.array([1, 0, 0])
        w3bhat = np.array([-1, 0, 0])
        w1 = self.max_frequency*w1hat
        w2 = self.max_frequency*w2hat
        w3 = self.max_frequency*w3hat
        w3b = self.max_frequency*w3bhat
        
        # Calculate the Theta value for each magnet relative to the object
        theta1 = np.arctan2(np.linalg.norm(np.cross(w1hat,p1)),np.dot(w1hat,p1))
        theta2 = np.arctan2(np.linalg.norm(np.cross(w2hat,p2)),np.dot(w2hat,p2))
        theta3 = np.arctan2(np.linalg.norm(np.cross(w3hat,p3)),np.dot(w3hat,p3))
        theta3b = np.arctan2(np.linalg.norm(np.cross(w3bhat,p3)),np.dot(w3bhat,p3))
        

        
        # Construct unit rho, theta, and phi vectors
        ip1 = p1/np.linalg.norm(p1)
        ip2 = p2/np.linalg.norm(p2)
        if Fdes[0] < 0:
            ip3 = ip2
        else:
            ip3 = ip1
        iphi1 = np.cross(w1hat,p1)/np.linalg.norm(np.cross(w1hat,p1))
        iphi2 = np.cross(w2hat,p2)/np.linalg.norm(np.cross(w2hat,p2))
        iphi3 = np.cross(w3hat,p3)/np.linalg.norm(np.cross(w3hat,p3))
        

        iphi3b = np.cross(w3bhat,p3)/np.linalg.norm(np.cross(w3bhat,p3))
        itheta1 = np.cross(iphi1,ip1)
        itheta2 = np.cross(iphi2,ip2)
        itheta3 = np.cross(iphi3,ip3)
        itheta3b = np.cross(iphi3b,ip3)
                
        
        # print('constnats')
        # r = self.model.object_properties.radius
        # sigma = self.model.object_properties.sigma
        
        # print(r,sigma)
        
        # c0p90,c1p90,c2p90,c3p90,c4p90 = self.model.cylinderModel.constants['frhorho']
        # print(c0p90,c1p90,c2p90,c3p90,c4p90)
        # c0p0,c1p0,c2p0,c3p0,c4p0 = self.model.cylinderModel.constants['fzz']
        # print(c0p0,c1p0,c2p0,c3p0,c4p0)
        # c0phi,c1phi,c2phi,c3phi,c4phi = self.model.cylinderModel.constants['frhophi']
        # print(c0phi,c1phi,c2phi,c3phi,c4phi)
        # c0tp,c1tp,c2tp,c3tp,c4tp =  self.model.cylinderModel.constants['tzz']
        # print(c0tp,c1tp,c2tp,c3tp,c4tp)
        # c0ttheta,c1ttheta,c2ttheta,c3ttheta,c4ttheta =  self.model.cylinderModel.constants['trhoz']
        # print(c0ttheta,c1ttheta,c2ttheta,c3ttheta,c4ttheta)
        
        
        
        # # Calculate the non-dimensional terms
        # Pi1_1 = sigma*self.mu0*np.linalg.norm(w1)*r**2
        # Pi1_2 = sigma*self.mu0*np.linalg.norm(w2)*r**2
        # Pi1_3 = sigma*self.mu0*np.linalg.norm(w3)*r**2
        # Pi2_1 = np.linalg.norm(p1)/r
        # Pi2_2 = np.linalg.norm(p2)/r
        # Pi2_3 = np.linalg.norm(p3)/r
        # Pi0_1p90 = ((c0p90*Pi1_1)**(c1p90*Pi1_1**c2p90)*10**c3p90)/Pi2_1**c4p90
        # Pi0_2p90 = ((c0p90*Pi1_2)**(c1p90*Pi1_2**c2p90)*10**c3p90)/Pi2_2**c4p90
        # Pi0_3p90 = ((c0p90*Pi1_3)**(c1p90*Pi1_3**c2p90)*10**c3p90)/Pi2_3**c4p90
        # Pi0_1p0 = ((c0p0*Pi1_1)**(c1p0*Pi1_1**c2p0)*10**c3p0)/Pi2_1**c4p0
        # Pi0_2p0 = ((c0p0*Pi1_2)**(c1p0*Pi1_2**c2p0)*10**c3p0)/Pi2_2**c4p0
        # Pi0_3p0 = ((c0p0*Pi1_3)**(c1p0*Pi1_3**c2p0)*10**c3p0)/Pi2_3**c4p0
        # Pi0_1phi90 = ((c0phi*Pi1_1)**(c1phi*Pi1_1**c2phi)*10**c3phi)/Pi2_1**c4phi
        # Pi0_2phi90 = ((c0phi*Pi1_2)**(c1phi*Pi1_2**c2phi)*10**c3phi)/Pi2_2**c4phi
        # Pi0_1tp = ((c0tp*Pi1_1)**(c1tp*Pi1_1**c2tp)*10**c3tp)/Pi2_1**c4tp
        # Pi0_2tp = ((c0tp*Pi1_2)**(c1tp*Pi1_2**c2tp)*10**c3tp)/Pi2_2**c4tp
        # Pi0_3tp = ((c0tp*Pi1_3)**(c1tp*Pi1_3**c2tp)*10**c3tp)/Pi2_3**c4tp
        # Pi0_1ttheta = ((c0ttheta*Pi1_1)**(c1ttheta*Pi1_1**c2ttheta)*10**c3ttheta)/Pi2_1**c4ttheta
        # Pi0_2ttheta = ((c0ttheta*Pi1_2)**(c1ttheta*Pi1_2**c2ttheta)*10**c3ttheta)/Pi2_2**c4ttheta
        # Pi0_3ttheta = ((c0ttheta*Pi1_3)**(c1ttheta*Pi1_3**c2ttheta)*10**c3ttheta)/Pi2_3**c4ttheta
        
        
        # # Calculate the p=0 p=90 rho and phi forces
        # f1_p90 = Pi0_1p90*self.mu0/r**4
        # f2_p90 = Pi0_2p90*self.mu0/r**4
        # f3_p90 = Pi0_3p90*self.mu0/r**4
        # f1_p0 = Pi0_1p0*self.mu0/r**4
        # f2_p0 = Pi0_2p0*self.mu0/r**4
        # f3_p0 = Pi0_3p0*self.mu0/r**4
        # f1_phi90 = Pi0_1phi90*self.mu0/r**4
        # f2_phi90 = Pi0_2phi90*self.mu0/r**4
        # t1_p0 = Pi0_1tp*self.mu0/r**3
        # t2_p0 = Pi0_2tp*self.mu0/r**3
        # t3_p0 = Pi0_3tp*self.mu0/r**3
        # t1_theta90 = Pi0_1ttheta*self.mu0/r**3
        # t2_theta90 = Pi0_2ttheta*self.mu0/r**3
        # t3_theta90 = Pi0_3ttheta*self.mu0/r**3
        
        # print(f1_p0,t1_p0,t1_theta90,f1_phi90)


        #can just get these values from model        
        f1_p0,t1_p0,f1_p90,f1_phi90,t1_theta90 = self.model.sphereModel.modelCall(dist = np.linalg.norm(p1),dipole_strength = 1)
        
        f2_p0,t2_p0,f2_p90,f2_phi90,t2_theta90 = self.model.sphereModel.modelCall(dist = np.linalg.norm(p2),dipole_strength = 1)

        f3_p0,t3_p0,f3_p90,f3_phi90,t3_theta90 = self.model.sphereModel.modelCall(dist = np.linalg.norm(p3),dipole_strength = 1)
        

        # Calculate the Forces at theta location
        f1_p = -(f1_p90 - f1_p0)/2*np.cos(2*theta1) + (f1_p90 + f1_p0)/2
        f2_p = -(f2_p90 - f2_p0)/2*np.cos(2*theta2) + (f2_p90 + f2_p0)/2
        f3_p = -(f3_p90 - f3_p0)/2*np.cos(2*theta3) + (f3_p90 + f3_p0)/2
        f3b_p = -(f3_p90 - f3_p0)/2*np.cos(2*theta3b) + (f3_p90 + f3_p0)/2
        f1_phi = f1_phi90*np.sin(theta1)
        f2_phi = f2_phi90*np.sin(theta2)
        t1_p = t1_p0*np.cos(theta1)
        t2_p = t2_p0*np.cos(theta2)
        t3_p = t3_p0*np.cos(theta3)
        t3b_p = t3_p0*np.cos(theta3b)
        t1_theta = t1_theta90*np.sin(theta1)
        t2_theta = t2_theta90*np.sin(theta2)
        t3_theta = t3_theta90*np.sin(theta3)
        t3b_theta = t3_theta90*np.sin(theta3b)
        
        # Calculate the total force from each omnimagnet
        if np.isnan(iphi1).any():
            f1 = (f1_p*np.transpose(ip1))* self.max_dipole_strength**2
        else:
            f1 = (f1_p*np.transpose(ip1) + f1_phi*np.transpose(iphi1))* self.max_dipole_strength**2
        if np.isnan(iphi1).any():
            f2 = (f2_p*np.transpose(ip2))* self.max_dipole_strength**2
        else:
            f2 = (f2_p*np.transpose(ip2) + f2_phi*np.transpose(iphi2))* self.max_dipole_strength**2
        f3 = f3_p*np.transpose(ip3)* self.max_dipole_strength**2
        f3b = f3b_p*np.transpose(ip3)* self.max_dipole_strength**2
        if np.isnan(itheta1).any():
            t1 = (t1_p*np.transpose(ip1))* self.max_dipole_strength**2
        else:
            t1 = (t1_p*np.transpose(ip1) + t1_theta*np.transpose(itheta1))* self.max_dipole_strength**2
        if np.isnan(itheta2).any():
            t2 = (t2_p*np.transpose(ip2))* self.max_dipole_strength**2
        else:
            t2 = (t2_p*np.transpose(ip2) + t2_theta*np.transpose(itheta2))* self.max_dipole_strength**2
        if np.isnan(itheta3).any():
            t3 = (t3_p*np.transpose(ip3))* self.max_dipole_strength**2
        else:
            t3 = (t3_p*np.transpose(ip3) + t3_theta*np.transpose(itheta3))* self.max_dipole_strength**2
        if np.isnan(itheta3b).any():
            t3b = (t3b_p*np.transpose(ip3))* self.max_dipole_strength**2
        else:
            t3b = (t3b_p*np.transpose(ip3) + t3b_theta*np.transpose(itheta3b))* self.max_dipole_strength**2
        
        
        #this is identical for f1,f2,f3, not sure what f3b is supposed to be but you can figure that out and delete a lot of code
        print('wrenches')
        print(f1,f2,f3)
        
        #this function expects a numpy array of dipole strengths, psi+xi angles, the model values and the rho vector
        mag1_wrench = modelCallRotatedWrench(np.array(self.max_dipole_strength).reshape(1),deconstruct_omega(w1hat),(f1_p0,t1_p0,f1_p90,f1_phi90,t1_theta90),p1.reshape((3,1)))
        mag2_wrench = modelCallRotatedWrench(np.array(self.max_dipole_strength).reshape(1),deconstruct_omega(w2hat),(f2_p0,t2_p0,f2_p90,f2_phi90,t2_theta90),p2.reshape((3,1)))
        mag3_wrench = modelCallRotatedWrench(np.array(self.max_dipole_strength).reshape(1),deconstruct_omega(w3hat),(f3_p0,t3_p0,f3_p90,f3_phi90,t3_theta90),p3.reshape((3,1)))
        f1 = mag1_wrench[0,:3,0]
        f2 = mag2_wrench[0,:3,0]
        f3 = mag3_wrench[0,:3,0]
        print(f1,f2,f3)
        
        Fyz = np.linalg.norm(f1 + f2)
        Fx = np.linalg.norm(f3)
        d_x = (fxd*Fyz)/(fyzd*Fx + fxd*Fyz)
        d_yz = (1-d_x)
        Fmax = f3*d_x/2 + f3b*d_x/2 + (f1 + f2)*d_yz
        Tmax = t3*d_x/2 + t3b*d_x/2 + (t1 + t2)*d_yz

        
        
        # Manipulate mx and myz to get force magnitude correct if desired force 
        # magnitude is possible else give max force values    
        if np.linalg.norm(Fdes) > np.linalg.norm(Fmax):
            return Fmax, Tmax, w1, w2, w3, w3b,  self.max_dipole_strength,  self.max_dipole_strength,  self.max_dipole_strength,  self.max_dipole_strength, d_yz/2, d_yz/2, d_x/2, d_x/2
        else:
            Fratio = np.linalg.norm(Fdes)/np.linalg.norm(Fmax);
            m2ratio = np.sqrt(Fratio);
            m1_set = self.max_dipole_strength*m2ratio;
            m2_set = self.max_dipole_strength*m2ratio;
            m3_set = self.max_dipole_strength*m2ratio;
            if np.isnan(iphi1).any():
                f1 = (f1_p*np.transpose(ip1))*m1_set**2
            else:
                f1 = (f1_p*np.transpose(ip1) + f1_phi*np.transpose(iphi1))*m1_set**2
            if np.isnan(iphi1).any():
                f2 = (f2_p*np.transpose(ip2))*m2_set**2
            else:
                f2 = (f2_p*np.transpose(ip2) + f2_phi*np.transpose(iphi2))*m2_set**2
            f3 = f3_p*np.transpose(ip3)*m3_set**2
            f3b = f3b_p*np.transpose(ip3)*m3_set**2
            if np.isnan(itheta1).any():
                t1 = (t1_p*np.transpose(ip1))*m1_set**2
            else:
                t1 = (t1_p*np.transpose(ip1) + t1_theta*np.transpose(itheta1))*m1_set**2
            if np.isnan(itheta2).any():
                t2 = (t2_p*np.transpose(ip2))*m2_set**2
            else:
                t2 = (t2_p*np.transpose(ip2) + t2_theta*np.transpose(itheta2))*m2_set**2
            if np.isnan(itheta3).any():
                t3 = (t3_p*np.transpose(ip3))*m3_set**2
            else:
                t3 = (t3_p*np.transpose(ip3) + t3_theta*np.transpose(itheta3))*m3_set**2
            if np.isnan(itheta3b).any():
                t3b = (t3b_p*np.transpose(ip3))*m3_set**2
            else:
                t3b = (t3b_p*np.transpose(ip3) + t3b_theta*np.transpose(itheta3b))*m3_set**2
             
            F = f3*d_x/2 + f3b*d_x/2 + (f1 + f2)*d_yz;
            T = t3*d_x/2 + t3b*d_x/2 + (t1 + t2)*d_yz;
            return F, T, w1, w2, w3, w3b, m1_set, m2_set, m3_set, m3_set, d_yz/2, d_yz/2, d_x/2, d_x/2


        
    def solveWrench(self,position,desired_wrench):
        start = time.time()
        model_id =   self.count % (2 * len(self.model.magnets))    
        self.count += 1
        if model_id == 0:
            F,T,w1, w2, w3, w3b, m1_set, m2_set, m3_set, m3_set, d_yz, _, d_x, _ = self.calculate_duty_cycling(position, desired_wrench)
            print(m1_set, m2_set, m3_set, m3_set, d_yz, d_x)
            
            
            #rescale everything so a magnet is on for increments of self.duty_percent of the time
            scale_factor_t1 = 1 
            scale_factor_t2 = 1 
            
            if np.isnan(d_x) or np.isnan(d_yz) or d_x==0.0 or d_yz==0.0:
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
            
            for i in range(2 * len(self.model.magnets)):
                angles = deconstruct_omega(self.omegas[i])
                control = SphereControlCommand(mag_num = i%2,
                                               dipole_strength=self.dipoles[i],
                                               psi=angles[0],xi=angles[1])
                if i <=1:
                    control.seconds = self.T1 * self.duty_time
                else:
                    control.seconds = self.T2 * self.duty_time
                
                actual_wrench += self.model.forwardSphere(position,control)*control.seconds
                
                self.controls.append(control)

        control = self.controls[model_id]

        
        #print('chosen control',control)
        self.time += time.time()-start
        return control