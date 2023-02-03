#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 13:52:59 2023

@author: tabor
"""
states = controls = env = i = None
import sys

sys.path.append("..")  # Adds higher directory to python modules path.
import dataloader
from models import SphereControlCommand
from scipy.spatial.transform import Rotation as R
from ObjectProperties import ObjectProperties, compute_boat_properties

import numpy as np


def extract_sys_id_data(file_name, alt_rotational_velocity=False, skip_end=0):
    dataloader.load(file_name, globals())
    global controls, mass, i, numSteps, radius, object_properties_approximate, initial_object_properties, sigma_copper, boat_properties

    transform_modify_function = lambda x: x

    if type(controls[0]) == list:
        controls = [
            SphereControlCommand(
                mag_num=control[0],
                dipole_strength=control[1],
                psi=control[2],
                xi=control[3],
            )
            for control in controls
        ]
        boat_mass, boat_MOI = compute_boat_properties(0.059, 0.075, 0.07, 0.031)
        boat_properties = ObjectProperties(
            mass=boat_mass, MOI=boat_MOI
        )  # + ObjectProperties(0.15,np.zeros(3))
        mass, MOI = mass
        initial_object_properties = ObjectProperties(
            mass, MOI, radius * 5, 58000000.0 * 5
        )

        transform_modify_function = np.linalg.inv

    try:
        num_steps = numSteps
    except:
        num_steps = i

    try:
        initial_object_properties
    except:
        initial_object_properties = object_properties_approximate
    boat = boat_properties
    velocity_shift = 0
    start = -num_steps + 5
    end = -5 - skip_end

    if velocity_shift > 0:
        end -= velocity_shift
    else:
        start += velocity_shift

    # FIX incorrect velocity in real data
    linear_velocity = np.array(
        [
            transform_modify_function(state[0][:3, :3]) @ state[2][:, 0]
            for state in states[1 + start + velocity_shift : end - 1 + velocity_shift]
        ]
    )

    rotational_velocity = np.array(
        [
            transform_modify_function(state[0][:3, :3]) @ state[1][:, 0]
            for state in states[1 + start + velocity_shift : end - 1 + velocity_shift]
        ]
    )

    if alt_rotational_velocity:
        angles = np.array(
            [
                R.from_matrix(state[0][:3, :3]).as_euler("xyz")
                for state in states[1 + start : end]
            ]
        )
        rotational_velocity = []
        for current, future in zip(angles[:-1, 2], angles[1:, 2]):
            if np.abs(current - future) > np.pi:
                current -= np.sign(current) * 2 * np.pi
            vel = future - current
            rotational_velocity.append([0, 0, vel])
        rotational_velocity = np.array(rotational_velocity)

    positions = np.array([state[0][:3, 3] for state in states[1 + start : end - 1]])
    selected_controls = controls[start + 1 : end - 1]

    return (
        linear_velocity,
        rotational_velocity,
        positions,
        selected_controls,
        initial_object_properties,
        boat,
        env,
    )
