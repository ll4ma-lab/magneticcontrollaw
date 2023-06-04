# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:17:57 2023

@author: devin
"""

import numpy as np
import math

def ForceFinder(w,m,material,FEA_Exp,r,lambda1):

    mu0 = 4*math.pi*10**-7
    m_loc_start = np.array([[lambda1, 0, 0]])
    obj_loc_start = np.array([0, 0, 0])


    # Determine Object's mass and sigma
    if material == 'aluminum':
        sigma = 3.69e7;
    elif material == 'copper':
        sigma = 5.87e7;
    elif material == 'titanium':
        sigma = 2.4e6;


    # Define Model Constants       
    if FEA_Exp == 1:
        c0p90 = 266
        c1p90 = 2.6
        c2p90 = -0.101
        c3p90 = -7.65
        c4p90 = 7
        c0p0 = 460
        c1p0 = 2.95
        c2p0 = -0.101
        c3p0 = -9.26
        c4p0 = 7
        c0phi = 6040
        c1phi = 3.45
        c2phi = -0.102
        c3phi = -14.3
        c4phi = 7
        c0tp = 6840
        c1tp = 3.0
        c2tp = -0.0986
        c3tp = -13.2
        c4tp = 6
        c0ttheta = 8100
        c1ttheta = 3.6
        c2ttheta = -0.0985
        c3ttheta = -15.7
        c4ttheta = 6
    elif FEA_Exp == 0:
        c0p90 = 282
        c1p90 = 3.2
        c2p90 = -0.098
        c3p90 = -9.41
        c4p90 = 7
        c0p0 = 467
        c1p0 = 2.81
        c2p0 = -0.0969
        c3p0 = -9.75
        c4p0 = 7
        c0phi = 5870
        c1phi = 3.49
        c2phi = -0.0973
        c3phi = -14.6
        c4phi = 7
        c0tp = 6900
        c1tp = 3.35
        c2tp = -0.099
        c3tp = 14.9
        c4tp = 6
        c0ttheta = 8000
        c1ttheta = 3.4
        c2ttheta = -0.0928
        c3ttheta = -15
        c4ttheta = 6
     
    what = w/np.linalg.norm(w)
    p = obj_loc_start - m_loc_start

    # Calculate the Theta value for each magnet relative to the object
    if np.linalg.norm(w) == w[0]:
        theta = 0
    elif np.linalg.norm(w) == -w[0]:
        theta = math.pi
    else:           
        theta = math.atan2(np.linalg.norm(np.cross(what,p)),np.dot(what,np.transpose(p)))
        
    # Construct unit rho, theta, and phi vectors
    ip = p/np.linalg.norm(p);
    if np.linalg.norm(np.cross(what,p)) != 0:
        iphi = np.cross(what,p)/np.linalg.norm(np.cross(what,p));
    else:
        iphi = float("nan")
    if np.isnan(iphi).any():
        itheta = float("nan")
    else:
        itheta = np.cross(iphi,ip);
        
    # Calculate the non-dimensional terms
    Pi1 = sigma*mu0*np.linalg.norm(w)*r**2
    Pi2 = np.linalg.norm(p)/r
    Pi0_p90 = ((c0p90*Pi1)**(c1p90*Pi1**c2p90)*10**c3p90)/Pi2**c4p90
    Pi0_p0 = ((c0p0*Pi1)**(c1p0*Pi1**c2p0)*10**c3p0)/Pi2**c4p0
    Pi0_phi900 = ((c0phi*Pi1)**(c1phi*Pi1**c2phi)*10**c3phi)/Pi2**c4phi
    Pi0_tp = ((c0tp*Pi1)**(c1tp*Pi1**c2tp)*10**c3tp)/Pi2**c4tp;
    Pi0_ttheta = ((c0ttheta*Pi1)**(c1ttheta*Pi1**c2ttheta)*10**c3ttheta)/Pi2**c4ttheta;
    
    
    
    # Calculate the p=0 p=90 rho and phi forces
    f_p90 = Pi0_p90*mu0/r**4
    f_p0 = Pi0_p0*mu0/r**4
    f_phi90 = Pi0_phi900*mu0/r**4
    t_p0 = Pi0_tp*mu0/r**3;
    t_theta90 = Pi0_ttheta*mu0/r**3;
    
    
    # Calculate the Forces at theta location
    f_p = -(f_p90 - f_p0)/2*math.cos(2*theta) + (f_p90 + f_p0)/2
    f_phi = f_phi90*math.sin(theta)
    t_p = t_p0*math.cos(theta);
    t_theta = t_theta90*math.sin(theta);
    
    # Calculate the total force from each omnimagnet
    if np.isnan(iphi).any():
        f = (f_p*np.transpose(ip))*m**2;
    else:
        f = (f_p*np.transpose(ip) + f_phi*np.transpose(iphi))*m**2;
    if np.isnan(itheta).any():
        t = (t_p*np.transpose(ip))*m**2;
    else:
        t = (t_p*np.transpose(ip) + t_theta*np.transpose(itheta))*m**2;


    
    return f, abs(f_phi)/abs(f_p), t

class define_object:
    def __init__(self, state=np.array([[0.03, 0.03, 0.03],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]), material='al', radius=0.02):
        self.material = material
        self.radius = radius
        self.start_state = state
        if self.material == 'al':
            self.el_conductivity = 3.77e07
            self.density = 2710
        elif self.material == 'cu':
            self.el.conductivity = 5.87e07
            self.density = 8940
        self.mass = 4/3*math.pi*self.radius**3*self.density

def min_torque_solver(self, mags, position, fdes):
    
    # Determine the position vectors/unit vectors/normal unit vectors
    p1 = position - mags.position[0]
    p2 = position - mags.position[1]


    if fdes[0] < 0:
        p3 = p1
        mag3 = 0
        mag4 = 0
    else:
        p3 = p2
        mag3 = 1
        mag4 = 1

    # Determine the parrallel and perpendicular forces and the rotation about
    # the x-axis
    fxd = abs(fdes[0])
    fyzd = np.sqrt(fdes[1] ** 2 + fdes[2] ** 2)

    # Determine the direction for omega 1 and 2 based on fdes_perp
    ix = np.array([1, 0, 0])
    # fdes_parallel = np.array([fxd, 0, 0])
    fdes_perp = np.array([0, fdes[1], fdes[2]])
    fhat_des_perp = fdes_perp / np.linalg.norm(fdes_perp)
    w1hat = np.cross(fhat_des_perp,ix)
    w2hat = np.cross(ix,fhat_des_perp)
    w3hat = np.array([1, 0, 0])
    w3bhat = np.array([-1, 0, 0])
    w1 = mags.max_frequency * w1hat
    w2 = mags.max_frequency * w2hat
    w3 = mags.max_frequency * w3hat
    w3b = mags.max_frequency * w3bhat

    # Calculate the Theta value for each magnet relative to the object
    theta1 = np.arctan2(np.linalg.norm(np.cross(w1hat, p1)), np.dot(w1hat, p1))
    theta2 = np.arctan2(np.linalg.norm(np.cross(w2hat, p2)), np.dot(w2hat, p2))
    theta3 = np.arctan2(np.linalg.norm(np.cross(w3hat, p3)), np.dot(w3hat, p3))
    theta3b = np.arctan2(np.linalg.norm(np.cross(w3bhat, p3)), np.dot(w3bhat, p3))

    # Construct unit rho, theta, and phi vectors
    ip1 = p1 / np.linalg.norm(p1)
    ip2 = p2 / np.linalg.norm(p2)
    if fdes[0] < 0:
        ip3 = ip1
    else:
        ip3 = ip2
    iphi1 = np.cross(w1hat, p1) / np.linalg.norm(np.cross(w1hat, p1))
    iphi2 = np.cross(w2hat, p2) / np.linalg.norm(np.cross(w2hat, p2))
    iphi3 = np.cross(w3hat, p3) / np.linalg.norm(np.cross(w3hat, p3))
    iphi3b = np.cross(w3bhat, p3) / np.linalg.norm(np.cross(w3bhat, p3))
    itheta1 = np.cross(iphi1, ip1)
    itheta2 = np.cross(iphi2, ip2)
    itheta3 = np.cross(iphi3, ip3)
    itheta3b = np.cross(iphi3b, ip3)

    # Find the total force from each magnet
    f1 = ForceFinder(w1,mags.maxdipole,sphere.object,0,sphere.radius,)

    # this function expects a numpy array of dipole strengths, psi+xi angles,
    # the model values and the rho vector

    mag1_wrench = model_call_rotated_wrench(
        np.array(self.max_dipole_strength).reshape(1),
        deconstruct_omega(w1hat),
        (f1_p0, t1_p0, f1_p90, f1_phi90, t1_theta90),
        p1.reshape((3, 1)),
    )
    mag2_wrench = model_call_rotated_wrench(
        np.array(self.max_dipole_strength).reshape(1),
        deconstruct_omega(w2hat),
        (f2_p0, t2_p0, f2_p90, f2_phi90, t2_theta90),
        p2.reshape((3, 1)),
    )
    mag3_wrench = model_call_rotated_wrench(
        np.array(self.max_dipole_strength).reshape(1),
        deconstruct_omega(w3hat),
        (f3_p0, t3_p0, f3_p90, f3_phi90, t3_theta90),
        p3.reshape((3, 1)),
    )
    f1 = mag1_wrench[0, :3, 0]
    f2 = mag2_wrench[0, :3, 0]
    f3 = mag3_wrench[0, :3, 0]
    # print(f1,f2,f3)

    fyz = np.linalg.norm(f1 + f2)
    fx = np.linalg.norm(f3)
    d_x = (fxd * fyz) / (fyzd * fx + fxd * fyz)
    d_yz = 1 - d_x
    fmax = f3 * d_x / 2 + f3b * d_x / 2 + (f1 + f2) * d_yz
    tmax = t3 * d_x / 2 + t3b * d_x / 2 + (t1 + t2) * d_yz

    # Manipulate mx and myz to get force magnitude correct if desired force
    # magnitude is possible else give max force values
    if np.linalg.norm(fdes) > np.linalg.norm(fmax):
        return (
            fmax,
            tmax,
            w1,
            w2,
            w3,
            w3b,
            0,
            1,
            mag3,
            mag4,
            self.max_dipole_strength,
            self.max_dipole_strength,
            self.max_dipole_strength,
            self.max_dipole_strength,
            d_yz / 2,
            d_yz / 2,
            d_x / 2,
            d_x / 2,
        )
    else:
        fratio = np.linalg.norm(fdes) / np.linalg.norm(fmax)
        m2ratio = np.sqrt(fratio)
        m1_set = self.max_dipole_strength * m2ratio
        m2_set = self.max_dipole_strength * m2ratio
        m3_set = self.max_dipole_strength * m2ratio
        if np.isnan(iphi1).any():
            f1 = (f1_p * np.transpose(ip1)) * m1_set**2
        else:
            f1 = (
                f1_p * np.transpose(ip1) + f1_phi * np.transpose(iphi1)
            ) * m1_set**2
        if np.isnan(iphi1).any():
            f2 = (f2_p * np.transpose(ip2)) * m2_set**2
        else:
            f2 = (
                f2_p * np.transpose(ip2) + f2_phi * np.transpose(iphi2)
            ) * m2_set**2
        f3 = f3_p * np.transpose(ip3) * m3_set**2
        f3b = f3b_p * np.transpose(ip3) * m3_set**2
        if np.isnan(itheta1).any():
            t1 = (t1_p * np.transpose(ip1)) * m1_set**2
        else:
            t1 = (
                t1_p * np.transpose(ip1) + t1_theta * np.transpose(itheta1)
            ) * m1_set**2
        if np.isnan(itheta2).any():
            t2 = (t2_p * np.transpose(ip2)) * m2_set**2
        else:
            t2 = (
                t2_p * np.transpose(ip2) + t2_theta * np.transpose(itheta2)
            ) * m2_set**2
        if np.isnan(itheta3).any():
            t3 = (t3_p * np.transpose(ip3)) * m3_set**2
        else:
            t3 = (
                t3_p * np.transpose(ip3) + t3_theta * np.transpose(itheta3)
            ) * m3_set**2
        if np.isnan(itheta3b).any():
            t3b = (t3b_p * np.transpose(ip3)) * m3_set**2
        else:
            t3b = (
                t3b_p * np.transpose(ip3) + t3b_theta * np.transpose(itheta3b)
            ) * m3_set**2

        F = f3 * d_x / 2 + f3b * d_x / 2 + (f1 + f2) * d_yz
        T = t3 * d_x / 2 + t3b * d_x / 2 + (t1 + t2) * d_yz
        return (
            F,
            T,
            w1,
            w2,
            w3,
            w3b,
            0,
            1,
            mag3,
            mag4,
            m1_set,
            m2_set,
            m3_set,
            m3_set,
            d_yz / 2,
            d_yz / 2,
            d_x / 2,
            d_x / 2,
        )


    # Get the positions from magnets to object
    p1, p2 = -1 * (np.array(self.model.magnets) - np.transpose(position))
    if fdes[0] < 0:
        mag = 0
        mag2 = 1
    else:
        mag = 1
        mag2 = 0

    # Define FEA or Exp coefficients (1 = FEA, O = Exp) and initial search
    # parameters and precisions
    material = "copper"
    lambda1 = np.linalg.norm(p1)
    mmax = self.max_dipole_strength
    wmag = self.max_frequency
    r = self.model.object_properties.radius
    fea__exp = 0
    dtheta = math.pi / 20
    dtheta_precision = 1e-8
    dd = 0.1
    dd_precision = 1e-8

    # Handle the special case where thes desired force is entirely in the
    # x-direction
    ix = np.array([1, 0, 0])
    fdes_perp = np.array([0, fdes[1], fdes[2]])
    if np.linalg.norm(fdes_perp) == 0:
        w1hat = np.array([1, 0, 0])
        w1 = wmag * w1hat
        if fdes[0] < 0:
            F, fratio, T = force_finder(w1, mmax, material, fea__exp, r, lambda1)
        else:
            F, fratio, T = force_finder(w1, mmax, material, fea__exp, r, -lambda1)
        if np.linalg.norm(fdes) >= np.linalg.norm(F):
            return F, T, w1, -w1, mag, mag, mmax, mmax, 0.5, 0.5
        else:
            fmax_ratio = np.linalg.norm(fdes) / np.linalg.norm(F)
            mratio = math.sqrt(fmax_ratio)
            m1_set = mratio * mmax
            if fdes[0] < 0:
                F, fratio, T = force_finder(
                    w1, m1_set, material, fea__exp, r, lambda1
                )
            else:
                F, fratio, T = force_finder(
                    w1, m1_set, material, fea__exp, r, -lambda1
                )
            return F, T, w1, -w1, mag, mag, m1_set, m1_set, 0.5, 0.5

    # Now handle all other cases
    else:
        fhat_des_perp = fdes_perp / np.linalg.norm(fdes_perp)
        if fdes[0] >= 0:
            w1hat = np.cross(ix, fhat_des_perp)
        elif fdes[0] < 0:
            w1hat = np.cross(fhat_des_perp, ix)
        else:
            w1hat = np.cross(fhat_des_perp, ix)
        w1 = wmag * w1hat

        if fdes[0] < 0:
            F, fratio, T = force_finder(w1, mmax, material, fea__exp, r, lambda1)
        else:
            F, fratio, T = force_finder(w1, mmax, material, fea__exp, r, -lambda1)

        # Determine if desired force can be created using a single rotating dipole
        fdes_ratio = np.linalg.norm(fdes_perp) / abs(fdes[0])
        if fdes_ratio <= fratio:
            theta = math.pi / 2
            k1 = fhat_des_perp[0]
            k2 = fhat_des_perp[1]
            k3 = fhat_des_perp[2]

            while dtheta > dtheta_precision:

                vt = 1 - math.cos(math.pi / 2 - theta)
                ct = math.cos(math.pi / 2 - theta)
                st = math.sin(math.pi / 2 - theta)

                # Use angle-axis theorem to rotate about fhat_des_perp by theta
                R = np.array(
                    [
                        [
                            k1**2 * vt + ct,
                            k1 * k2 * vt - k3 * st,
                            k1 * k3 * vt + k2 * st,
                        ],
                        [
                            k1 * k2 * vt + k3 * st,
                            k2**2 * vt + ct,
                            k2 * k3 * vt - k1 * st,
                        ],
                        [
                            k1 * k3 * vt - k2 * st,
                            k2 * k3 * vt + k1 * st,
                            k3**2 * vt + ct,
                        ],
                    ]
                )

                # ry = np.array(
                #     [
                #         [
                #             math.cos(math.pi / 2 - theta),
                #             0,
                #             math.sin(math.pi / 2 - theta),
                #         ],
                #         [0, 1, 0],
                #         [
                #             -math.sin(math.pi / 2 - theta),
                #             0,
                #             math.cos(math.pi / 2 - theta),
                #         ],
                #     ]
                # )

                w1hatb = np.matmul(R, w1hat)
                w1b = wmag * w1hatb
                w1b2 = np.array([-1, 1, 1]) * w1b

                if fdes[0] < 0:
                    f1, fratio1, t1 = force_finder(
                        w1b, mmax, material, fea__exp, r, lambda1
                    )
                    f2, fratio2, t2 = force_finder(
                        w1b2, mmax, material, fea__exp, r, lambda1
                    )
                else:
                    f1, fratio1, t1 = force_finder(
                        w1b, mmax, material, fea__exp, r, -lambda1
                    )
                    f2, fratio2, t2 = force_finder(
                        w1b2, mmax, material, fea__exp, r, -lambda1
                    )
                F = (f1 + f2) / 2
                T = (t1 + t2) / 2
                fratio = (fratio1 + fratio2) / 2

                if fdes_ratio == fratio:
                    dtheta = dtheta_precision
                elif fratio > fdes_ratio:
                    theta = theta - dtheta
                    if theta < 0:
                        theta = theta + dtheta * 1.5
                        dtheta = dtheta / 2
                else:
                    dtheta = dtheta / 2
                    theta = theta + dtheta

            if np.linalg.norm(fdes) >= np.linalg.norm(F):
                return F, T, w1b, w1b2, mag, mag, mmax, mmax, 0.5, 0.5
            else:
                fmax_ratio = np.linalg.norm(fdes) / np.linalg.norm(F)
                mratio = math.sqrt(fmax_ratio)
                m1_set = mratio * mmax
                if fdes[0] < 0:
                    f1, fratio1, t2 = force_finder(
                        w1b, m1_set, material, fea__exp, r, lambda1
                    )
                    f2, fratio2, t2 = force_finder(
                        w1b2, m1_set, material, fea__exp, r, lambda1
                    )
                else:
                    f1, fratio1, t2 = force_finder(
                        w1b, m1_set, material, fea__exp, r, -lambda1
                    )
                    f2, fratio2, t2 = force_finder(
                        w1b2, m1_set, material, fea__exp, r, -lambda1
                    )
                return (
                    (f1 + f2) / 2,
                    (t1 + t2) / 2,
                    w1b,
                    w1b2,
                    mag,
                    mag,
                    m1_set,
                    m1_set,
                    0.5,
                    0.5,
                )

        # Now all forces not within the cone of one rotating dipole field
        else:

            w2hat = w1hat
            w2hat[1] = -w2hat[1]
            w2hat[2] = -w2hat[2]
            w2 = wmag * w2hat

            # Treat Fdes in perp direction as special case
            if fdes[0] == 0:
                d1 = 0.5
                d2 = 1 - d1
                f1, fratio1, t1 = force_finder(
                    w1, mmax, material, fea__exp, r, -lambda1
                )
                f2, fratio2, t2 = force_finder(
                    w2, mmax, material, fea__exp, r, lambda1
                )
                F = d1 * f1 + d2 * f2
                T = d1 * t1 + d2 * t2
                if np.linalg.norm(fdes) >= np.linalg.norm(F):
                    return F, T, w1, w2, mag, mag2, mmax, mmax, 0.5, 0.5
                else:
                    fmax_ratio = np.linalg.norm(fdes) / np.linalg.norm(F)
                    mratio = math.sqrt(fmax_ratio)
                    m1_set = mratio * mmax
                    f1, fratio1, t1 = force_finder(
                        w1, m1_set, material, fea__exp, r, -lambda1
                    )
                    f2, fratio2, t2 = force_finder(
                        w2, m1_set, material, fea__exp, r, lambda1
                    )
                    F = f1 * d1 + f2 * d2
                    T = t1 * d1 + t2 * d2
                    return F, T, w1, w2, mag, mag2, m1_set, m1_set, 0.5, 0.5

            else:

                d1 = 0.6
                d2 = 1 - d1

                while dd > dd_precision:
                    if fdes[0] < 0:
                        f1, fratio1, t1 = force_finder(
                            w1, mmax, material, fea__exp, r, lambda1
                        )
                        f2, fratio2, t2 = force_finder(
                            w2, mmax, material, fea__exp, r, -lambda1
                        )
                    else:
                        f1, fratio1, t1 = force_finder(
                            w1, mmax, material, fea__exp, r, -lambda1
                        )
                        f2, fratio2, t2 = force_finder(
                            w2, mmax, material, fea__exp, r, lambda1
                        )

                    F = d1 * f1 + d2 * f2
                    fratio = math.sqrt(F[2] ** 2 + F[1] ** 2) / abs(F[0])

                    if fdes_ratio == fratio:
                        dd = dd_precision
                    elif fratio > fdes_ratio:
                        d1 = d1 + dd
                        d2 = d2 - dd
                    else:
                        dd = dd / 2
                        d1 = d1 - dd
                        d2 = d2 + dd

                if np.linalg.norm(fdes) >= np.linalg.norm(F):
                    T = d1 * t1 + d2 * t2
                    return F, T, w1, w2, mag, mag2, mmax, mmax, d1, d2
                else:
                    fmax_ratio = np.linalg.norm(fdes) / np.linalg.norm(F)
                    mratio = math.sqrt(fmax_ratio)
                    m1_set = mratio * mmax
                    if fdes[0] < 0:
                        f1, fratio1, t1 = force_finder(
                            w1, m1_set, material, fea__exp, r, lambda1
                        )
                        f2, fratio2, t2 = force_finder(
                            w2, m1_set, material, fea__exp, r, -lambda1
                        )
                    else:
                        f1, fratio1, t1 = force_finder(
                            w1, m1_set, material, fea__exp, r, -lambda1
                        )
                        f2, fratio2, t2 = force_finder(
                            w2, m1_set, material, fea__exp, r, lambda1
                        )
                    F = d1 * f1 + d2 * f2
                    T = d1 * t1 + d2 * t2
                    return F, T, w1, w2, mag, mag2, m1_set, m1_set, d1, d2

# Experiment Setup

class mags:
    def __init__(self,  magdistance = .2225, max_frequency = 15, max_dipole = 40):
        self.position = np.array([[magdistance, 0, 0],[-magdistance, 0, 0]])
        self.max_frequency = max_frequency
        self.max_dipole = max_dipole
        
alpha = 0.5
my_mags = mags()
solver = 'min_torque_solver'
sphere = define_object()
p_des = np.array([0,0,0])
tolerance = 0.0001

if solver == 'min_torque_solver':
    duty_cycle_time = 4


# Define gains
if solver == 'min_torque_solver':
    T = 4
elif solver == 'max_force_solver':
    T = 2
kp = (sphere.mass)/(T**2)*(alpha**2 -2*alpha + 1)
kd = (4*(sphere.mass)*(1-alpha) - kp*T**2)/(2*T)


# Start Simulation
p_current = sphere.start_state[0]
p_dot_current = sphere.start_state[1]
r_vel_current = sphere.start_state[2]
while np.linalg.norm(p_des - p_current) > tolerance:
    
    
    # Calculate Desired Force
    f_des = kp*(p_des - p_current) + kd*p_dot_current



