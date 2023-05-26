# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:27:35 2019

@author: tabor
"""
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

if __name__ == "__main__":

    use_special_case_solver = False
    use_sphere = True
    cheat = False

    run_adaptive = True and use_sphere and not cheat
    inverse_dynamics = True and use_sphere and not cheat and not use_special_case_solver

    simulate_boat = False

    controller_knows_boat = False and simulate_boat

    vel_noise = False
    control_noise = False

    num_magnets = 2
    env = SpaceProblem(SpaceEnvironments.REGULATOR, num_magnets)
    # env.magnets = env.magnets[:4]
    #env.trajectoryDimensions = 'xyz'

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
    initial_radius_guess = 0.01
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

    if use_special_case_solver:
        solver = AxisAlignedSolver(solver_model, env.trajectory_dimensions)
        solver = DevinSolver(solver_model)
    else:
        solver = MultiMagSolver(
            model=solver_model,
            state_cost_string=env.trajectory_dimensions,
            object_properties=object_properties_approximate,
            sphere_model=use_sphere,
            inverse_dynamics=inverse_dynamics,
            boat=controller_boat,
            num_processes=10,
        )

    if inverse_dynamics:
        controller = PDController(env.start_state, 0.012, 0.004, 0.012, 0.013, 0, 0)
    elif use_special_case_solver:
        controller = PDController(env.start_state, 0.0005, 0.0, 0.1, 0.0, 0.0, 0.000)
    else:
        controller = PDController(
            env.start_state, 0.0005, 0.000001, 0.0005, 0.000002, 0.0, 0.000
        )
    simulator = DiscreteTimeSimulator(env.start_state, object_properties)
    velocity_profile = SplineProfile.SplineProfile(env, env.times)
    adaptive = Adaptive(
        env.magnets,
        "all",
        object_properties_approximate,
        25,
        adaptive_model,
        boat_properties,
    )

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
    )

    # print('Time spent doing forwad model ' + str(model.time))
    # print('Time spent doing solving model ' + str(solver.time))
    # print('Time spent doing velocity ' + str(velocityProfile.time))
    # print('Time spent doing simulation ' + str(simulator.time))
    vis_start = time.time()
    positions = np.array([state[0][:3, 3] for state in states])
    linear_velocity = np.array(
        [np.dot(state[0][:3, :3], state[2][:, 0]) for state in states]
    )
    rotational_velocity = np.array(
        [np.dot(state[0][:3, :3], state[1][:, 0]) for state in states]
    )
    transforms = np.array([state[0] for state in states])
    desired_final_tf = build_tf(desired_states[-1][:3, 0], desired_states[-1][3:6, 0])
    remaining_error = np.array(
        [controller.compute_error(state[0], desired_final_tf) for state in states]
    )

    # visualization stuff
    if run_adaptive:
        object_parameters_over_time = [(adaptive.parameters[0][:2].T @ np.diag((1,adaptive.problem.sigma_multiplier))).T] \
                                      * (sim_time - len(adaptive.parameters))\
                                      + adaptive.parameters
    else:
        object_parameters_over_time = [[object_properties_approximate.radius,object_properties_approximate.sigma]] * sim_time

       
    env.radius *= 0.5
    anim = visualization.animate_2d_path(
        env,
        positions,
        transforms,
        controls,
        object_parameters_over_time=object_parameters_over_time,
        name="",
        save_plot_steps=velocity_profile.switching_times,
        save_location=False,
    )
    npdesired_states = np.array(desired_states)

    # plots = {
    #     "Position (m)": np.transpose(positions),
    #     "Velocity (m/s)": np.transpose(linear_velocity),
    #     "Rot velocity (_/s)": np.transpose(rotational_velocity),
    #     "Force ": np.transpose(np.array(wrenches)[:, :3, 0]),
    #     "Torque ": np.transpose(np.array(wrenches)[:, 3:, 0]),
    #     "Desired position (m)": np.transpose(npdesired_states[:, :3, 0]),
    #     "Desired velocity (m/s) ": np.transpose(npdesired_states[:, 6:9, 0]),
    # }

    # visualization.plot_stuff(plots, "Controller")

    plots = {
        "Position Error(m)": np.abs(np.transpose(remaining_error[:, :3, 0])),
        "Orientation Error(radians)": -np.transpose(remaining_error[:, 3:6, 0]),
    }

    visualization.plot_stuff(plots, "Error")

    # plots = {
    #     "desired Position(m)": np.array(desired_states)[:, :3, 0].T,
    #     "desired Orientation (radians)": np.array(desired_states)[:, 3:6, 0].T,
    # }

    # visualization.plot_stuff(plots, "Desired Pose")

    # # print('Time spent doing visualization ' + str(time.time()-visStart))

    # plt.figure()
    # plt.plot(minimization_costs)

    # diff = npdesired_states[:, :3, 0] - positions[:, :]
    # plt.figure()
    # plt.plot(diff)
    # print(np.mean(np.square(diff)))
    # print(np.amax(diff))

    import dataloader
    from datetime import datetime
    
    if not use_special_case_solver:
        del solver.pool
    if run_adaptive:
        del adaptive.optimizer
    
    dataloader.save(
        "../data/sim_" + str(sim_time) + "s_" + str(datetime.now()).replace(":", "."),
        [globals()],
    )
