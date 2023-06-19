5# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 19:35:46 2022

@author: tabor
"""
import numpy as np
import time
import random
from models import SphereControlCommand, deconstruct_omega, model_call_rotated_wrench


class DevinSolver:
    def __init__(self, model):
        self.model = model
        self.count = 0
        self.time = 0
        self.outcount = 0

        self.max_frequency = 15
        self.max_dipole_strength = 40

        self.mu0 = 4 * np.pi * 10**-7

        self.duty_time = 4
        min_full_rotations = 3
        self.duty_percent = min_full_rotations / (self.max_frequency * self.duty_time)

    def calculate_duty_cycling(self, position, wrench):
        # Determine the position vectors/unit vectors/normal unit vectors
        p1, p2 = -1 * (np.array(self.model.magnets) - np.transpose(position))

        fdes = np.copy(wrench[:3, 0])

        # seemed like different coordinate system
        # Fdes[1]*=-1

        if fdes[0] < 0:
            p3 = p1
            mag3 = 0
            mag4 = 0
        else:
            p3 = p2
            mag3 = 1
            mag4 = 1

        # print('')
        # print('')
        # print('F desired')
        # print(Fdes)
        # print(' rho vectors')
        # print(p1,p2,p3)
        # print('magnets')
        # print(self.model.magnets)
        # print('')
        # print('')

        # Determine the parrallel and perpendicular forces and the rotation about
        # the x-axis
        fxd = abs(fdes[0])
        fyzd = np.sqrt(fdes[1] ** 2 + fdes[2] ** 2)

        # Determine the direction for omega 1 and 2 based on fyzd
        ix = np.array([1, 0, 0])
        # fdes_parallel = np.array([fxd, 0, 0])
        fdes_perp = np.array([0, fdes[1], fdes[2]])
        fhat_des_perp = fdes_perp / np.linalg.norm(fdes_perp)

        w1hat = np.cross(fhat_des_perp,ix)
        w2hat = np.cross(ix,fhat_des_perp)
        w3hat = np.array([1, 0, 0])
        w3bhat = np.array([-1, 0, 0])
        w1 = self.max_frequency * w1hat
        w2 = self.max_frequency * w2hat
        w3 = self.max_frequency * w3hat
        w3b = self.max_frequency * w3bhat

        # print('')
        # print('')
        # print('Fdes')
        # print(ix)
        # print(fdes_parallel)
        # print(fdes_perp)
        # print(fhat_des_perp)
        # print('omegas')
        # print(w1)
        # print(w2)
        # print(w3)
        # print(w3b)
        # print('')
        # print('')

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

        # can just get these values from model
        f1_p0, t1_p0, f1_p90, f1_phi90, t1_theta90 = self.model.sphere_model.model_call(
            dist=np.linalg.norm(p1), dipole_strength=1
        )

        f2_p0, t2_p0, f2_p90, f2_phi90, t2_theta90 = self.model.sphere_model.model_call(
            dist=np.linalg.norm(p2), dipole_strength=1
        )

        f3_p0, t3_p0, f3_p90, f3_phi90, t3_theta90 = self.model.sphere_model.model_call(
            dist=np.linalg.norm(p3), dipole_strength=1
        )

        # Calculate the Forces at theta location
        f1_p = -(f1_p90 - f1_p0) / 2 * np.cos(2 * theta1) + (f1_p90 + f1_p0) / 2
        f2_p = -(f2_p90 - f2_p0) / 2 * np.cos(2 * theta2) + (f2_p90 + f2_p0) / 2
        f3_p = -(f3_p90 - f3_p0) / 2 * np.cos(2 * theta3) + (f3_p90 + f3_p0) / 2
        f3b_p = -(f3_p90 - f3_p0) / 2 * np.cos(2 * theta3b) + (f3_p90 + f3_p0) / 2
        f1_phi = f1_phi90 * np.sin(theta1)
        f2_phi = f2_phi90 * np.sin(theta2)
        t1_p = t1_p0 * np.cos(theta1)
        t2_p = t2_p0 * np.cos(theta2)
        t3_p = t3_p0 * np.cos(theta3)
        t3b_p = t3_p0 * np.cos(theta3b)
        t1_theta = t1_theta90 * np.sin(theta1)
        t2_theta = t2_theta90 * np.sin(theta2)
        t3_theta = t3_theta90 * np.sin(theta3)
        t3b_theta = t3_theta90 * np.sin(theta3b)

        # Calculate the total force from each omnimagnet
        if np.isnan(iphi1).any():
            f1 = (f1_p * np.transpose(ip1)) * self.max_dipole_strength**2
        else:
            f1 = (
                f1_p * np.transpose(ip1) + f1_phi * np.transpose(iphi1)
            ) * self.max_dipole_strength**2
        if np.isnan(iphi1).any():
            f2 = (f2_p * np.transpose(ip2)) * self.max_dipole_strength**2
        else:
            f2 = (
                f2_p * np.transpose(ip2) + f2_phi * np.transpose(iphi2)
            ) * self.max_dipole_strength**2
        f3 = f3_p * np.transpose(ip3) * self.max_dipole_strength**2
        f3b = f3b_p * np.transpose(ip3) * self.max_dipole_strength**2
        if np.isnan(itheta1).any():
            t1 = (t1_p * np.transpose(ip1)) * self.max_dipole_strength**2
        else:
            t1 = (
                t1_p * np.transpose(ip1) + t1_theta * np.transpose(itheta1)
            ) * self.max_dipole_strength**2
        if np.isnan(itheta2).any():
            t2 = (t2_p * np.transpose(ip2)) * self.max_dipole_strength**2
        else:
            t2 = (
                t2_p * np.transpose(ip2) + t2_theta * np.transpose(itheta2)
            ) * self.max_dipole_strength**2
        if np.isnan(itheta3).any():
            t3 = (t3_p * np.transpose(ip3)) * self.max_dipole_strength**2
        else:
            t3 = (
                t3_p * np.transpose(ip3) + t3_theta * np.transpose(itheta3)
            ) * self.max_dipole_strength**2
        if np.isnan(itheta3b).any():
            t3b = (t3b_p * np.transpose(ip3)) * self.max_dipole_strength**2
        else:
            t3b = (
                t3b_p * np.transpose(ip3) + t3b_theta * np.transpose(itheta3b)
            ) * self.max_dipole_strength**2

        # this is identical for f1,f2,f3, not sure what f3b is supposed to be
        # but you can figure that out and delete a lot of code

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

    def solve_wrench(self, position, desired_wrench):
        start = time.time()
        model_id = self.count % (2 * len(self.model.magnets))
        switcher_flag = self.count % 4
        self.count += 1
        if self.count == 1:
            cointoss = random.random()
            if cointoss <= 0.5:
                if model_id == 0:
                    (
                        F,
                        T,
                        w1,
                        w2,
                        w3,
                        w3b,
                        mag1,
                        mag2,
                        mag3,
                        mag4,
                        m1_set,
                        m2_set,
                        m3_set,
                        m3_set,
                        d_yz,
                        _,
                        d_x,
                        _,
                    ) = self.calculate_duty_cycling(position, desired_wrench)
                    mags = [mag1, mag2, mag3, mag4]
            
                    # rescale everything so a magnet is on for increments of self.duty_percent of the time
                    scale_factor_t1 = 1
                    scale_factor_t2 = 1
            
                    if np.isnan(d_x) or np.isnan(d_yz) or d_x == 0.0 or d_yz == 0.0:
                        d_x = 0 if np.isnan(d_x) else d_x
                        d_yz = 0 if np.isnan(d_yz) else d_yz
                        self.t1 = d_yz
                        self.t2 = d_x
                    else:
                        # rescale smaller one down, then adjust accordingly
                        if d_yz < d_x:
                            self.t1 = self.duty_percent * int(d_yz / self.duty_percent)
                            self.t2 = 1 / 2 - self.t1
                            if self.t2 < 0.5:
                                scale_factor_t2 = np.sqrt((d_x * self.t1) / (d_yz * self.t2))
                        else:
                            self.t2 = self.duty_percent * int(d_x / self.duty_percent)
                            self.t1 = 1 / 2 - self.t2
                            if self.t1 < 0.5:
                                scale_factor_t1 = np.sqrt((d_yz * self.t2) / (d_x * self.t1))
            
                    dipole_1 = m1_set * scale_factor_t1
                    dipole_2 = m2_set * scale_factor_t1
                    dipole_3 = m3_set * scale_factor_t2
                    dipole_4 = m3_set * scale_factor_t2
            
                    self.dipoles = [dipole_1, dipole_2, dipole_3, dipole_4]
                    self.omegas = [
                        w1 / np.linalg.norm(w1),
                        w2 / np.linalg.norm(w2),
                        w3 / np.linalg.norm(w3),
                        w3b / np.linalg.norm(w3b),
                    ]
                    self.controls = []
            
                    actual_wrench = np.zeros((6, 1))
            
            
            
                    for i in range(2 * len(self.model.magnets)):
                        angles = deconstruct_omega(self.omegas[i])
            
                        control = SphereControlCommand(
                            mag_num=mags[i],
                            dipole_strength=self.dipoles[i],
                            psi=angles[0],
                            xi=angles[1],
                        )
                        if i <= 1:
                            control.seconds = self.t1 * self.duty_time
                        else:
                            control.seconds = self.t2 * self.duty_time
            
                        actual_wrench += (
                            self.model.forward_sphere(position, control) * control.seconds
                        )
                        # print(F*self.duty_time,T*self.duty_time)
                        self.controls.append(control)
            else:
                if model_id == 0:
                    (
                        F,
                        T,
                        w2,
                        w1,
                        w3b,
                        w3,
                        mag2,
                        mag1,
                        mag4,
                        mag3,
                        m1_set,
                        m2_set,
                        m3_set,
                        m3_set,
                        d_yz,
                        _,
                        d_x,
                        _,
                    ) = self.calculate_duty_cycling(position, desired_wrench)
                    mags = [mag1, mag2, mag3, mag4]
            
                    # rescale everything so a magnet is on for increments of self.duty_percent of the time
                    scale_factor_t1 = 1
                    scale_factor_t2 = 1
            
                    if np.isnan(d_x) or np.isnan(d_yz) or d_x == 0.0 or d_yz == 0.0:
                        d_x = 0 if np.isnan(d_x) else d_x
                        d_yz = 0 if np.isnan(d_yz) else d_yz
                        self.t1 = d_yz
                        self.t2 = d_x
                    else:
                        # rescale smaller one down, then adjust accordingly
                        if d_yz < d_x:
                            self.t1 = self.duty_percent * int(d_yz / self.duty_percent)
                            self.t2 = 1 / 2 - self.t1
                            if self.t2 < 0.5:
                                scale_factor_t2 = np.sqrt((d_x * self.t1) / (d_yz * self.t2))
                        else:
                            self.t2 = self.duty_percent * int(d_x / self.duty_percent)
                            self.t1 = 1 / 2 - self.t2
                            if self.t1 < 0.5:
                                scale_factor_t1 = np.sqrt((d_yz * self.t2) / (d_x * self.t1))
            
                    dipole_1 = m1_set * scale_factor_t1
                    dipole_2 = m2_set * scale_factor_t1
                    dipole_3 = m3_set * scale_factor_t2
                    dipole_4 = m3_set * scale_factor_t2
            
                    self.dipoles = [dipole_1, dipole_2, dipole_3, dipole_4]
                    self.omegas = [
                        w1 / np.linalg.norm(w1),
                        w2 / np.linalg.norm(w2),
                        w3 / np.linalg.norm(w3),
                        w3b / np.linalg.norm(w3b),
                    ]
                    self.controls = []
            
                    actual_wrench = np.zeros((6, 1))
            
            
            
                    for i in range(2 * len(self.model.magnets)):
                        angles = deconstruct_omega(self.omegas[i])
            
                        control = SphereControlCommand(
                            mag_num=mags[i],
                            dipole_strength=self.dipoles[i],
                            psi=angles[0],
                            xi=angles[1],
                        )
                        if i <= 1:
                            control.seconds = self.t1 * self.duty_time
                        else:
                            control.seconds = self.t2 * self.duty_time
            
                        actual_wrench += (
                            self.model.forward_sphere(position, control) * control.seconds
                        )
                        # print(F*self.duty_time,T*self.duty_time)
                        self.controls.append(control)
        else:
            
            if self.outcount % 2 == 0:
                if model_id == 0:
                    (
                        F,
                        T,
                        w1,
                        w2,
                        w3,
                        w3b,
                        mag1,
                        mag2,
                        mag3,
                        mag4,
                        m1_set,
                        m2_set,
                        m3_set,
                        m3_set,
                        d_yz,
                        _,
                        d_x,
                        _,
                    ) = self.calculate_duty_cycling(position, desired_wrench)
                    mags = [mag1, mag2, mag3, mag4]
            
                    # rescale everything so a magnet is on for increments of self.duty_percent of the time
                    scale_factor_t1 = 1
                    scale_factor_t2 = 1
            
                    if np.isnan(d_x) or np.isnan(d_yz) or d_x == 0.0 or d_yz == 0.0:
                        d_x = 0 if np.isnan(d_x) else d_x
                        d_yz = 0 if np.isnan(d_yz) else d_yz
                        self.t1 = d_yz
                        self.t2 = d_x
                    else:
                        # rescale smaller one down, then adjust accordingly
                        if d_yz < d_x:
                            self.t1 = self.duty_percent * int(d_yz / self.duty_percent)
                            self.t2 = 1 / 2 - self.t1
                            if self.t2 < 0.5:
                                scale_factor_t2 = np.sqrt((d_x * self.t1) / (d_yz * self.t2))
                        else:
                            self.t2 = self.duty_percent * int(d_x / self.duty_percent)
                            self.t1 = 1 / 2 - self.t2
                            if self.t1 < 0.5:
                                scale_factor_t1 = np.sqrt((d_yz * self.t2) / (d_x * self.t1))
            
                    dipole_1 = m1_set * scale_factor_t1
                    dipole_2 = m2_set * scale_factor_t1
                    dipole_3 = m3_set * scale_factor_t2
                    dipole_4 = m3_set * scale_factor_t2
            
                    self.dipoles = [dipole_1, dipole_2, dipole_3, dipole_4]
                    self.omegas = [
                        w1 / np.linalg.norm(w1),
                        w2 / np.linalg.norm(w2),
                        w3 / np.linalg.norm(w3),
                        w3b / np.linalg.norm(w3b),
                    ]
                    self.controls = []
            
                    actual_wrench = np.zeros((6, 1))
            
            
            
                    for i in range(2 * len(self.model.magnets)):
                        angles = deconstruct_omega(self.omegas[i])
            
                        control = SphereControlCommand(
                            mag_num=mags[i],
                            dipole_strength=self.dipoles[i],
                            psi=angles[0],
                            xi=angles[1],
                        )
                        if i <= 1:
                            control.seconds = self.t1 * self.duty_time
                        else:
                            control.seconds = self.t2 * self.duty_time
            
                        actual_wrench += (
                            self.model.forward_sphere(position, control) * control.seconds
                        )
                        # print(F*self.duty_time,T*self.duty_time)
                        self.controls.append(control)
            else:
                if model_id == 0:
                    (
                        F,
                        T,
                        w2,
                        w1,
                        w3b,
                        w3,
                        mag2,
                        mag1,
                        mag4,
                        mag3,
                        m1_set,
                        m2_set,
                        m3_set,
                        m3_set,
                        d_yz,
                        _,
                        d_x,
                        _,
                    ) = self.calculate_duty_cycling(position, desired_wrench)
                    mags = [mag1, mag2, mag3, mag4]
            
                    # rescale everything so a magnet is on for increments of self.duty_percent of the time
                    scale_factor_t1 = 1
                    scale_factor_t2 = 1
            
                    if np.isnan(d_x) or np.isnan(d_yz) or d_x == 0.0 or d_yz == 0.0:
                        d_x = 0 if np.isnan(d_x) else d_x
                        d_yz = 0 if np.isnan(d_yz) else d_yz
                        self.t1 = d_yz
                        self.t2 = d_x
                    else:
                        # rescale smaller one down, then adjust accordingly
                        if d_yz < d_x:
                            self.t1 = self.duty_percent * int(d_yz / self.duty_percent)
                            self.t2 = 1 / 2 - self.t1
                            if self.t2 < 0.5:
                                scale_factor_t2 = np.sqrt((d_x * self.t1) / (d_yz * self.t2))
                        else:
                            self.t2 = self.duty_percent * int(d_x / self.duty_percent)
                            self.t1 = 1 / 2 - self.t2
                            if self.t1 < 0.5:
                                scale_factor_t1 = np.sqrt((d_yz * self.t2) / (d_x * self.t1))
            
                    dipole_1 = m1_set * scale_factor_t1
                    dipole_2 = m2_set * scale_factor_t1
                    dipole_3 = m3_set * scale_factor_t2
                    dipole_4 = m3_set * scale_factor_t2
            
                    self.dipoles = [dipole_1, dipole_2, dipole_3, dipole_4]
                    self.omegas = [
                        w1 / np.linalg.norm(w1),
                        w2 / np.linalg.norm(w2),
                        w3 / np.linalg.norm(w3),
                        w3b / np.linalg.norm(w3b),
                    ]
                    self.controls = []
            
                    actual_wrench = np.zeros((6, 1))
            
            
            
                    for i in range(2 * len(self.model.magnets)):
                        angles = deconstruct_omega(self.omegas[i])
            
                        control = SphereControlCommand(
                            mag_num=mags[i],
                            dipole_strength=self.dipoles[i],
                            psi=angles[0],
                            xi=angles[1],
                        )
                        if i <= 1:
                            control.seconds = self.t1 * self.duty_time
                        else:
                            control.seconds = self.t2 * self.duty_time
            
                        actual_wrench += (
                            self.model.forward_sphere(position, control) * control.seconds
                        )
                        # print(F*self.duty_time,T*self.duty_time)
                        self.controls.append(control)
            if switcher_flag == 0:
                self.outcount += 1
        # print(m1_set, m2_set, m3_set, m3_set, d_yz, d_x)



        control = self.controls[model_id]

        # print('chosen control',control)
        self.time += time.time() - start
        return control
