# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:05:19 2024

@author: tabor
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:27:35 2019

@author: tabor
"""

import sys

sys.path.append("..")  # Adds higher directory to python modules path.
import numpy as np
from PDController import PDController
from discreteTimeSimulator import DiscreteTimeSimulator, simulate_experiment
from environments import (
    PlanarProblem,
    SpaceProblem,
    PlanarEnvironments,
    SpaceEnvironments,
)
from se3Utils import build_tf
import matplotlib.pyplot as plt
import time
import visualization
import DummyProfile
import SplineProfile
import TrapProfile
from Adaptive import Adaptive


from models import MultiMagModel
from solvers.modelOptimization import MultiMagSolver
from solvers.axisAlignedSolver import AxisAlignedSolver
from solvers.devinSolver import DevinSolver
from solvers.devinSolver2 import DevinSolver2

from ObjectProperties import (
    ObjectProperties,
    compute_boat_properties,
    compute_sphere_properties,
)
from sysIDoptimization import RadiusSysid, SeperateRadiusSysid, MassSysID
from tqdm import tqdm

if __name__ == "__main__":
    use_special_case_solver = False
    use_sphere = True
    cheat = False

    run_adaptive = False and use_sphere and not cheat
    inverse_dynamics = (
        False and use_sphere and not cheat and not use_special_case_solver
    )

    simulate_boat = False

    controller_knows_boat = False and simulate_boat

    vel_noise = False
    control_noise = False

    num_magnets = 2
    env = SpaceProblem(SpaceEnvironments.REGULATOR, num_magnets)
    env.times = [900,0]
    # env.magnets = env.magnets[:4]
    # env.trajectory_dimensions = 'xyz'

    sim_time = np.sum(env.times)

    sigma_copper = 5.87e07
    sigma_al = 3.77e07

    # simulator object
    radius = 0.02
    mass, MOI = compute_sphere_properties(8940, radius)
    object_properties = ObjectProperties(
        mass=mass, MOI=MOI, radius=radius, sigma=sigma_copper
    )

    # controller object
    initial_radius_guess = 0.02
    mass_initial, moi_initial = compute_sphere_properties(8940, radius)
    object_properties_approximate = ObjectProperties(
        mass=mass_initial,
        MOI=moi_initial,
        radius=initial_radius_guess,
        sigma=sigma_copper,
    )

    # boat object
    boat_mass, boat_moi = compute_boat_properties(0.059, 0.075, 0.07, 0.031)
    boat_properties = ObjectProperties(mass=boat_mass, MOI=boat_moi)
    # np.diag((0,1,0.5))@
    adaptive_model = RadiusSysid

    if simulate_boat:
        object_properties = object_properties + boat_properties
    if controller_knows_boat:
        controller_boat = boat_properties
    else:
        controller_boat = None

    model = MultiMagModel(
        env.magnets, object_properties, use_sphere
    )  # correct object properties for simulator
    solver_model = MultiMagModel(
        env.magnets, object_properties_approximate, use_sphere
    )  # approximate object properties for controller


    solver = MultiMagSolver(
        model=solver_model,
        state_cost_string=env.trajectory_dimensions,
        object_properties=object_properties_approximate,
        sphere_model=use_sphere,
        inverse_dynamics=inverse_dynamics,
        boat=controller_boat,
        num_processes=10,
    )


    controller = PDController(
        env.start_state, 0.0005, 0.000001, 0.0005, 0.000002, 0.0, 0.000
    )
    adaptive = Adaptive(
        env.magnets,
        "all",
        object_properties_approximate,
        25,
        adaptive_model,
        boat_properties,
    )



    all_states = []
    remaining_errors = []
    
    
    points = np.linspace(-0.04,0.04,10)
    starts = np.array(np.meshgrid(points,points,points)).reshape((3,-1)).T.reshape((-1,3,1))
    
    for start in tqdm(starts):
        env.start_state = np.concatenate((start,env.start_state[3:]))
        
        
        velocity_profile = SplineProfile.SplineProfile(env, env.times)

        simulator = DiscreteTimeSimulator(env.start_state, object_properties)
        (
            states,
            wrenches,
            desired_wrenchs,
            desired_states,
            controls,
            minimization_costs,
        ) = simulate_experiment(
            simulator,
            controller,
            env,
            velocity_profile,
            model,
            solver,
            adaptive,
            sim_time=sim_time,
            cheat=cheat,
            run_adaptive=run_adaptive,
            inverse_dynamics=inverse_dynamics,
            debug_prints=False
        )
    
        desired_final_tf = build_tf(desired_states[-1][:3, 0], desired_states[-1][3:6, 0])
        
        remaining_error = np.array(
            [controller.compute_error(state[0], desired_final_tf) for state in states]
        )
        
        all_states.append(states)
        remaining_errors.append(remaining_error)
        
    
   

    import dataloader
    from datetime import datetime

    if not use_special_case_solver:
        del solver.pool
    if run_adaptive:
        del adaptive.optimizer

    dataloader.save(
        "../../data/batch_" + str(controls[0]).split('(')[0] + "_" + str(num_magnets) +'magnetd_' + "s_" + str(datetime.now()).replace(":", "."),
        [globals()],
    )
