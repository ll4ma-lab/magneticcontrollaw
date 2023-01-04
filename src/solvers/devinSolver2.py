# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 19:35:46 2022

@author: tabor
"""
import numpy as np
import time
from models import SphereControlCommand, deconstruct_omega, modelCallRotatedWrench
from solvers.DevinHelper import ForceFinder
import math


class devinSolver2:
    def __init__(self, model):
        self.model = model
        self.count = 0
        self.time = 0

        self.max_frequency = 15
        self.max_dipole_strength = 40

        self.mu0 = 4 * np.pi * 10**-7

        self.duty_time = 2
        min_full_rotations = 3
        self.duty_percent = min_full_rotations / (self.max_frequency * self.duty_time)

    def calculate_duty_cycling_and_angles(self, position, wrench):

        # Grab the desired Force
        Fdes = np.copy(wrench[:3, 0])

        # Get the positions from magnets to object
        p1, p2 = -1 * (np.array(self.model.magnets) - np.transpose(position))
        if Fdes[0] < 0:
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
        FEA_Exp = 0
        dtheta = math.pi / 20
        dtheta_precision = 1e-8
        dd = 0.1
        dd_precision = 1e-8

        # Handle the special case where thes desired force is entirely in the
        # x-direction
        ix = np.array([1, 0, 0])
        fdes_perp = np.array([0, Fdes[1], Fdes[2]])
        if np.linalg.norm(fdes_perp) == 0:
            w1hat = np.array([1, 0, 0])
            w1 = wmag * w1hat
            if Fdes[0] < 0:
                F, Fratio, T = ForceFinder(w1, mmax, material, FEA_Exp, r, lambda1)
            else:
                F, Fratio, T = ForceFinder(w1, mmax, material, FEA_Exp, r, -lambda1)
            if np.linalg.norm(Fdes) >= np.linalg.norm(F):
                return F, T, w1, -w1, mag, mag, mmax, mmax, 0.5, 0.5
            else:
                Fmax_ratio = np.linalg.norm(Fdes) / np.linalg.norm(F)
                mratio = math.sqrt(Fmax_ratio)
                m1_set = mratio * mmax
                if Fdes[0] < 0:
                    F, Fratio, T = ForceFinder(
                        w1, m1_set, material, FEA_Exp, r, lambda1
                    )
                else:
                    F, Fratio, T = ForceFinder(
                        w1, m1_set, material, FEA_Exp, r, -lambda1
                    )
                return F, T, w1, -w1, mag, mag, m1_set, m1_set, 0.5, 0.5

        # Now handle all other cases
        else:
            fhat_des_perp = fdes_perp / np.linalg.norm(fdes_perp)
            if Fdes[0] >= 0:
                w1hat = np.cross(ix, fhat_des_perp)
            elif Fdes[0] < 0:
                w1hat = np.cross(fhat_des_perp, ix)
            w1 = wmag * w1hat

            if Fdes[0] < 0:
                F, Fratio, T = ForceFinder(w1, mmax, material, FEA_Exp, r, lambda1)
            else:
                F, Fratio, T = ForceFinder(w1, mmax, material, FEA_Exp, r, -lambda1)

            # Determine if desired force can be created using a single rotating dipole
            Fdes_ratio = np.linalg.norm(fdes_perp) / abs(Fdes[0])
            if Fdes_ratio <= Fratio:
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

                    Ry = np.array(
                        [
                            [
                                math.cos(math.pi / 2 - theta),
                                0,
                                math.sin(math.pi / 2 - theta),
                            ],
                            [0, 1, 0],
                            [
                                -math.sin(math.pi / 2 - theta),
                                0,
                                math.cos(math.pi / 2 - theta),
                            ],
                        ]
                    )

                    w1hatb = np.matmul(R, w1hat)
                    w1b = wmag * w1hatb
                    w1b2 = np.array([-1, 1, 1]) * w1b

                    if Fdes[0] < 0:
                        F1, Fratio1, T1 = ForceFinder(
                            w1b, mmax, material, FEA_Exp, r, lambda1
                        )
                        F2, Fratio2, T2 = ForceFinder(
                            w1b2, mmax, material, FEA_Exp, r, lambda1
                        )
                    else:
                        F1, Fratio1, T1 = ForceFinder(
                            w1b, mmax, material, FEA_Exp, r, -lambda1
                        )
                        F2, Fratio2, T2 = ForceFinder(
                            w1b2, mmax, material, FEA_Exp, r, -lambda1
                        )
                    F = (F1 + F2) / 2
                    T = (T1 + T2) / 2
                    Fratio = (Fratio1 + Fratio2) / 2

                    if Fdes_ratio == Fratio:
                        dtheta = dtheta_precision
                    elif Fratio > Fdes_ratio:
                        theta = theta - dtheta
                        if theta < 0:
                            theta = theta + dtheta * 1.5
                            dtheta = dtheta / 2
                    else:
                        dtheta = dtheta / 2
                        theta = theta + dtheta

                if np.linalg.norm(Fdes) >= np.linalg.norm(F):
                    return F, T, w1b, w1b2, mag, mag, mmax, mmax, 0.5, 0.5
                else:
                    Fmax_ratio = np.linalg.norm(Fdes) / np.linalg.norm(F)
                    mratio = math.sqrt(Fmax_ratio)
                    m1_set = mratio * mmax
                    if Fdes[0] < 0:
                        F1, Fratio1, T2 = ForceFinder(
                            w1b, m1_set, material, FEA_Exp, r, lambda1
                        )
                        F2, Fratio2, T2 = ForceFinder(
                            w1b2, m1_set, material, FEA_Exp, r, lambda1
                        )
                    else:
                        F1, Fratio1, T2 = ForceFinder(
                            w1b, m1_set, material, FEA_Exp, r, -lambda1
                        )
                        F2, Fratio2, T2 = ForceFinder(
                            w1b2, m1_set, material, FEA_Exp, r, -lambda1
                        )
                    return (
                        (F1 + F2) / 2,
                        (T1 + T2) / 2,
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
                if Fdes[0] == 0:
                    d1 = 0.5
                    d2 = 1 - d1
                    F1, Fratio1, T1 = ForceFinder(
                        w1, mmax, material, FEA_Exp, r, -lambda1
                    )
                    F2, Fratio2, T2 = ForceFinder(
                        w2, mmax, material, FEA_Exp, r, lambda1
                    )
                    F = d1 * F1 + d2 * F2
                    T = d1 * T1 + d2 * T2
                    if np.linalg.norm(Fdes) >= np.linalg.norm(F):
                        return F, T, w1, w2, mag, mag2, mmax, mmax, 0.5, 0.5
                    else:
                        Fmax_ratio = np.linalg.norm(Fdes) / np.linalg.norm(F)
                        mratio = math.sqrt(Fmax_ratio)
                        m1_set = mratio * mmax
                        F1, Fratio1, T1 = ForceFinder(
                            w1, m1_set, material, FEA_Exp, r, -lambda1
                        )
                        F2, Fratio2, T2 = ForceFinder(
                            w2, m1_set, material, FEA_Exp, r, lambda1
                        )
                        F = F1 * d1 + F2 * d2
                        T = T1 * d1 + T2 * d2
                        return F, T, w1, w2, mag, mag2, m1_set, m1_set, 0.5, 0.5

                else:

                    d1 = 0.6
                    d2 = 1 - d1

                    while dd > dd_precision:
                        if Fdes[0] < 0:
                            F1, Fratio1, T1 = ForceFinder(
                                w1, mmax, material, FEA_Exp, r, lambda1
                            )
                            F2, Fratio2, T2 = ForceFinder(
                                w2, mmax, material, FEA_Exp, r, -lambda1
                            )
                        else:
                            F1, Fratio1, T1 = ForceFinder(
                                w1, mmax, material, FEA_Exp, r, -lambda1
                            )
                            F2, Fratio2, T2 = ForceFinder(
                                w2, mmax, material, FEA_Exp, r, lambda1
                            )

                        F = d1 * F1 + d2 * F2
                        Fratio = math.sqrt(F[2] ** 2 + F[1] ** 2) / abs(F[0])

                        if Fdes_ratio == Fratio:
                            dd = dd_precision
                        elif Fratio > Fdes_ratio:
                            d1 = d1 + dd
                            d2 = d2 - dd
                        else:
                            dd = dd / 2
                            d1 = d1 - dd
                            d2 = d2 + dd

                    if np.linalg.norm(Fdes) >= np.linalg.norm(F):
                        T = d1 * T1 + d2 * T2
                        return F, T, w1, w2, mag, mag2, mmax, mmax, d1, d2
                    else:
                        Fmax_ratio = np.linalg.norm(Fdes) / np.linalg.norm(F)
                        mratio = math.sqrt(Fmax_ratio)
                        m1_set = mratio * mmax
                        if Fdes[0] < 0:
                            F1, Fratio1, T1 = ForceFinder(
                                w1, m1_set, material, FEA_Exp, r, lambda1
                            )
                            F2, Fratio2, T2 = ForceFinder(
                                w2, m1_set, material, FEA_Exp, r, -lambda1
                            )
                        else:
                            F1, Fratio1, T1 = ForceFinder(
                                w1, m1_set, material, FEA_Exp, r, -lambda1
                            )
                            F2, Fratio2, T2 = ForceFinder(
                                w2, m1_set, material, FEA_Exp, r, lambda1
                            )
                        F = d1 * F1 + d2 * F2
                        T = d1 * T1 + d2 * T2
                        return F, T, w1, w2, mag, mag2, m1_set, m1_set, d1, d2

    def solveWrench(self, position, desired_wrench):
        start = time.time()
        model_id = self.count % (len(self.model.magnets))
        self.count += 1
        if model_id == 0:
            (
                F,
                T,
                w1,
                w2,
                mag1,
                mag2,
                m1_set,
                m2_set,
                d1,
                d2,
            ) = self.calculate_duty_cycling_and_angles(position, desired_wrench)
            # print(m1_set, m2_set, m3_set, m3_set, d_yz, d_x)
            mags = [mag1, mag2]

            # rescale everything so a magnet is on for increments of self.duty_percent of the time
            scale_factor_t1 = 1
            scale_factor_t2 = 1

            """if np.isnan(d1) or np.isnan(d2):
                d_x = 0 if np.isnan(d_x) else d_x
                d_yz = 0 if np.isnan(d_yz) else d_yz
                self.T1 = d_yz
                self.T2 = d_x
            else:
                #rescale smaller one down, then adjust accordingly
                """
            if d1 < d2:
                self.T1 = self.duty_percent * int(d1 / self.duty_percent)
                self.T2 = 1 - self.T1
                if self.T2 < 0.5:
                    scale_factor_t2 = np.sqrt((d2 * self.T1) / (d1 * self.T2))
            else:
                self.T2 = self.duty_percent * int(d2 / self.duty_percent)
                self.T1 = 1 - self.T2
                if self.T1 < 0.5:
                    scale_factor_t1 = np.sqrt((d1 * self.T2) / (d2 * self.T1))

            dipole_1 = m1_set * scale_factor_t1
            dipole_2 = m2_set * scale_factor_t2

            self.dipoles = [dipole_1, dipole_2]
            self.omegas = [w1 / np.linalg.norm(w1), w2 / np.linalg.norm(w2)]
            self.controls = []

            actual_wrench = np.zeros((6, 1))

            print("")
            print("")
            print("Times")
            print(self.T1)
            print(self.T2)
            print("")
            print("")

            for i in range(len(self.model.magnets)):
                angles = deconstruct_omega(self.omegas[i])

                control = SphereControlCommand(
                    mag_num=mags[i],
                    dipole_strength=self.dipoles[i],
                    psi=angles[0],
                    xi=angles[1],
                )
                if i == 0:
                    control.seconds = self.T1 * self.duty_time
                else:
                    control.seconds = self.T2 * self.duty_time

                actual_wrench += (
                    self.model.forwardSphere(position, control) * control.seconds
                )
                # print(F*self.duty_time,T*self.duty_time)
                self.controls.append(control)
            print("")
            print("")
            print(actual_wrench / 2)
            print(desired_wrench)

            print("")
            print("")

        control = self.controls[model_id]

        # print('chosen control',control)
        self.time += time.time() - start
        return control
