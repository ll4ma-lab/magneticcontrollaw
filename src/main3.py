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


    for i in range(29,42):
        if i == 0:
            special_case_solver = 'DevinSolver'
            ex_filename =     'excelfiles/ds1_c1.csv'
            vel_ex_filename = 'excelfiles/ds1_c1_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR2)
            use_special_case_solver = True
        elif i == 1:
            special_case_solver = 'DevinSolver'
            ex_filename =     'excelfiles/ds1_c2.csv'
            vel_ex_filename = 'excelfiles/ds1_c2_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR3)
            use_special_case_solver = True
        elif i == 2:
            special_case_solver = 'DevinSolver'
            ex_filename =     'excelfiles/ds1_c3.csv'
            vel_ex_filename = 'excelfiles/ds1_c3_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR4)
            use_special_case_solver = True
        elif i == 3:
            special_case_solver = 'DevinSolver'
            ex_filename =     'excelfiles/ds1_c4.csv'
            vel_ex_filename = 'excelfiles/ds1_c4_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR5)
            use_special_case_solver = True
        elif i == 4:
            special_case_solver = 'DevinSolver'
            ex_filename =     'excelfiles/ds1_c5.csv'
            vel_ex_filename = 'excelfiles/ds1_c5_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR6)
            use_special_case_solver = True
        elif i == 5:
            special_case_solver = 'DevinSolver'
            ex_filename =     'excelfiles/ds1_c6.csv'
            vel_ex_filename = 'excelfiles/ds1_c6_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR7)
            use_special_case_solver = True
        elif i == 6:
            special_case_solver = 'DevinSolver'
            ex_filename =     'excelfiles/ds1_c7.csv'
            vel_ex_filename = 'excelfiles/ds1_c7_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR8)
            use_special_case_solver = True
        elif i == 7:
            special_case_solver = 'DevinSolver'
            ex_filename =     'excelfiles/ds1_c8.csv'
            vel_ex_filename = 'excelfiles/ds1_c8_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR9)
            use_special_case_solver = True
        elif i == 8:
            special_case_solver = 'DevinSolver'
            ex_filename =     'excelfiles/ds1_f1.csv'
            vel_ex_filename = 'excelfiles/ds1_f1_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR10)
            use_special_case_solver = True
        elif i == 9:
            special_case_solver = 'DevinSolver'
            ex_filename =     'excelfiles/ds1_f2.csv'
            vel_ex_filename = 'excelfiles/ds1_f2_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR11)
            use_special_case_solver = True
        elif i == 10:
            special_case_solver = 'DevinSolver'
            ex_filename =     'excelfiles/ds1_f3.csv'
            vel_ex_filename = 'excelfiles/ds1_f3_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR12)
            use_special_case_solver = True
        elif i == 11:
            special_case_solver = 'DevinSolver'
            ex_filename =     'excelfiles/ds1_f4.csv'
            vel_ex_filename = 'excelfiles/ds1_f4_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR13)
            use_special_case_solver = True
        elif i == 12:
            special_case_solver = 'DevinSolver'
            ex_filename =     'excelfiles/ds1_f5.csv'
            vel_ex_filename = 'excelfiles/ds1_f5_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR14)
            use_special_case_solver = True
        elif i == 13:
            special_case_solver = 'DevinSolver'
            ex_filename =     'excelfiles/ds1_f6.csv'
            vel_ex_filename = 'excelfiles/ds1_f6_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR15)
            use_special_case_solver = True
        elif i == 14:
            special_case_solver = 'DevinSolver2'
            ex_filename =     'excelfiles/ds2_c1.csv'
            vel_ex_filename = 'excelfiles/ds2_c1_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR2)
            use_special_case_solver = True
        elif i == 15:
            special_case_solver = 'DevinSolver2'
            ex_filename =     'excelfiles/ds2_c2.csv'
            vel_ex_filename = 'excelfiles/ds2_c2_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR3)
            use_special_case_solver = True
        elif i == 16:
            special_case_solver = 'DevinSolver2'
            ex_filename =     'excelfiles/ds2_c3.csv'
            vel_ex_filename = 'excelfiles/ds2_c3_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR4)
            use_special_case_solver = True
        elif i == 17:
            special_case_solver = 'DevinSolver2'
            ex_filename =     'excelfiles/ds2_c4.csv'
            vel_ex_filename = 'excelfiles/ds2_c4_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR5)
            use_special_case_solver = True
        elif i == 18:
            special_case_solver = 'DevinSolver2'
            ex_filename =     'excelfiles/ds2_c5.csv'
            vel_ex_filename = 'excelfiles/ds2_c5_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR6)
            use_special_case_solver = True
        elif i == 19:
            special_case_solver = 'DevinSolver2'
            ex_filename =     'excelfiles/ds2_c6.csv'
            vel_ex_filename = 'excelfiles/ds2_c6_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR7)
            use_special_case_solver = True
        elif i == 20:
            special_case_solver = 'DevinSolver2'
            ex_filename =     'excelfiles/ds2_c7.csv'
            vel_ex_filename = 'excelfiles/ds2_c7_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR8)
            use_special_case_solver = True
        elif i == 21:
            special_case_solver = 'DevinSolver2'
            ex_filename =     'excelfiles/ds2_c8.csv'
            vel_ex_filename = 'excelfiles/ds2_c8_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR9)
            use_special_case_solver = True
        elif i == 22:
            special_case_solver = 'DevinSolver2'
            ex_filename =     'excelfiles/ds2_f1.csv'
            vel_ex_filename = 'excelfiles/ds2_f1_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR10)
            use_special_case_solver = True
        elif i == 23:
            special_case_solver = 'DevinSolver2'
            ex_filename =     'excelfiles/ds2_f2.csv'
            vel_ex_filename = 'excelfiles/ds2_f2_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR11)
            use_special_case_solver = True
        elif i == 24:
            special_case_solver = 'DevinSolver2'
            ex_filename =     'excelfiles/ds2_f3.csv'
            vel_ex_filename = 'excelfiles/ds2_f3_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR12)
            use_special_case_solver = True
        elif i == 25:
            special_case_solver = 'DevinSolver2'
            ex_filename =     'excelfiles/ds2_f4.csv'
            vel_ex_filename = 'excelfiles/ds2_f4_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR13)
            use_special_case_solver = True
        elif i == 26:
            special_case_solver = 'DevinSolver2'
            ex_filename =     'excelfiles/ds2_f5.csv'
            vel_ex_filename = 'excelfiles/ds2_f5_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR14)
            use_special_case_solver = True
        elif i == 27:
            special_case_solver = 'DevinSolver2'
            ex_filename =     'excelfiles/ds2_f6.csv'
            vel_ex_filename = 'excelfiles/ds2_f6_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR15)
            use_special_case_solver = True
        elif i == 28:
            ex_filename =     'excelfiles/g_c1.csv'
            vel_ex_filename = 'excelfiles/g_c1_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR2)
            use_special_case_solver = False
            env.trajectory_dimensions = 'xyz'
            special_case_solver = 'griffin'
        elif i == 29:
            ex_filename =     'excelfiles/g_c2.csv'
            vel_ex_filename = 'excelfiles/g_c2_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR3)
            use_special_case_solver = False
            env.trajectory_dimensions = 'xyz'
            special_case_solver = 'griffin'
        elif i == 30:
            ex_filename =     'excelfiles/g_c3.csv'
            vel_ex_filename = 'excelfiles/g_c3_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR4)
            use_special_case_solver = False
            env.trajectory_dimensions = 'xyz'
            special_case_solver = 'griffin'
        elif i == 31:
            ex_filename =     'excelfiles/g_c4.csv'
            vel_ex_filename = 'excelfiles/g_c4_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR5)
            use_special_case_solver = False
            env.trajectory_dimensions = 'xyz'
            special_case_solver = 'griffin'
        elif i == 32:
            ex_filename =     'excelfiles/g_c5.csv'
            vel_ex_filename = 'excelfiles/g_c5_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR6)
            use_special_case_solver = False
            env.trajectory_dimensions = 'xyz'
            special_case_solver = 'griffin'
        elif i == 33:
            ex_filename =     'excelfiles/g_c6.csv'
            vel_ex_filename = 'excelfiles/g_c6_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR7)
            use_special_case_solver = False
            env.trajectory_dimensions = 'xyz'
            special_case_solver = 'griffin'
        elif i == 34:
            ex_filename =     'excelfiles/g_c7.csv'
            vel_ex_filename = 'excelfiles/g_c7_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR8)
            use_special_case_solver = False
            env.trajectory_dimensions = 'xyz'
            special_case_solver = 'griffin'
        elif i == 35:
            ex_filename =     'excelfiles/g_c8.csv'
            vel_ex_filename = 'excelfiles/g_c8_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR9)
            use_special_case_solver = False
            env.trajectory_dimensions = 'xyz'
            special_case_solver = 'griffin'
        elif i == 36:
            ex_filename =     'excelfiles/g_f1.csv'
            vel_ex_filename = 'excelfiles/g_f1_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR10)
            use_special_case_solver = False
            env.trajectory_dimensions = 'xyz'
            special_case_solver = 'griffin'
        elif i == 37:
            ex_filename =     'excelfiles/g_f2.csv'
            vel_ex_filename = 'excelfiles/g_f2_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR11)
            use_special_case_solver = False
            env.trajectory_dimensions = 'xyz'
            special_case_solver = 'griffin'
        elif i == 38:
            ex_filename =     'excelfiles/g_f3.csv'
            vel_ex_filename = 'excelfiles/g_f3_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR12)
            use_special_case_solver = False
            env.trajectory_dimensions = 'xyz'
            special_case_solver = 'griffin'
        elif i == 39:
            ex_filename =     'excelfiles/g_f4.csv'
            vel_ex_filename = 'excelfiles/g_f4_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR13)
            use_special_case_solver = False
            env.trajectory_dimensions = 'xyz'
            special_case_solver = 'griffin'
        elif i == 40:
            ex_filename =     'excelfiles/g_f5.csv'
            vel_ex_filename = 'excelfiles/g_f5_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR14)
            use_special_case_solver = False
            env.trajectory_dimensions = 'xyz'
            special_case_solver = 'griffin'
        elif i == 41:
            ex_filename =     'excelfiles/g_f6.csv'
            vel_ex_filename = 'excelfiles/g_f6_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR15)
            use_special_case_solver = False
            env.trajectory_dimensions = 'xyz'
            special_case_solver = 'griffin'
        elif i == 42:
            ex_filename =     'excelfiles/g2_c1.csv'
            vel_ex_filename = 'excelfiles/g2_c1_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR2)
            use_special_case_solver = False
            special_case_solver = 'griffin'
        elif i == 43:
            ex_filename =     'excelfiles/g2_c2.csv'
            vel_ex_filename = 'excelfiles/g2_c2_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR3)
            use_special_case_solver = False
            special_case_solver = 'griffin'
        elif i == 44:
            ex_filename =     'excelfiles/g2_c3.csv'
            vel_ex_filename = 'excelfiles/g2_c3_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR4)
            use_special_case_solver = False
            special_case_solver = 'griffin'
        elif i == 45:
            ex_filename =     'excelfiles/g2_c4.csv'
            vel_ex_filename = 'excelfiles/g2_c4_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR5)
            use_special_case_solver = False
            special_case_solver = 'griffin'
        elif i == 46:
            ex_filename =     'excelfiles/g2_c5.csv'
            vel_ex_filename = 'excelfiles/g2_c5_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR6)
            use_special_case_solver = False
            special_case_solver = 'griffin'
        elif i == 47:
            ex_filename =     'excelfiles/g2_c6.csv'
            vel_ex_filename = 'excelfiles/g2_c6_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR7)
            use_special_case_solver = False
            special_case_solver = 'griffin'
        elif i == 48:
            ex_filename =     'excelfiles/g2_c7.csv'
            vel_ex_filename = 'excelfiles/g2_c7_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR8)
            use_special_case_solver = False
            special_case_solver = 'griffin'
        elif i == 49:
            ex_filename =     'excelfiles/g2_c8.csv'
            vel_ex_filename = 'excelfiles/g2_c8_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR9)
            use_special_case_solver = False
            special_case_solver = 'griffin'
        elif i == 50:
            ex_filename =     'excelfiles/g2_f1.csv'
            vel_ex_filename = 'excelfiles/g2_f1_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR10)
            use_special_case_solver = False
            special_case_solver = 'griffin'
        elif i == 51:
            ex_filename =     'excelfiles/g2_f2.csv'
            vel_ex_filename = 'excelfiles/g2_f2_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR11)
            use_special_case_solver = False
            special_case_solver = 'griffin'
        elif i == 52:
            ex_filename =     'excelfiles/g2_f3.csv'
            vel_ex_filename = 'excelfiles/g2_f3_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR12)
            use_special_case_solver = False
            special_case_solver = 'griffin'
        elif i == 53:
            ex_filename =     'excelfiles/g2_f4.csv'
            vel_ex_filename = 'excelfiles/g2_f4_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR13)
            use_special_case_solver = False
            special_case_solver = 'griffin'
        elif i == 54:
            ex_filename =     'excelfiles/g2_f5.csv'
            vel_ex_filename = 'excelfiles/g2_f5_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR14)
            use_special_case_solver = False
            special_case_solver = 'griffin'
        elif i == 55:
            ex_filename =     'excelfiles/g2_f6.csv'
            vel_ex_filename = 'excelfiles/g2_f6_r.csv'
            env = SpaceProblem(SpaceEnvironments.REGULATOR15)
            use_special_case_solver = False
            special_case_solver = 'griffin'

        

        use_sphere = True
        cheat = False
    
        run_adaptive = False and use_sphere
        inverse_dynamics = True and use_sphere and not cheat and not use_special_case_solver
    
        simulate_boat = False
    
        controller_knows_boat = True and simulate_boat
    
        vel_noise = False
        control_noise = False
    
        env.magnets = env.magnets[:]

    
        sim_time = 997#np.sum(env.times)
    
        sigma_copper = 5.87e07
        sigma_al = 3.77e07
    
        # simulator object
        radius = 0.02
        #mass, MOI = compute_sphere_properties(8940, radius)
        mass, MOI = compute_sphere_properties(2710, radius)
        object_properties = ObjectProperties(
            mass=mass, MOI=MOI, radius=radius, sigma=sigma_al
        )
    
          # controller object
        initial_radius_guess = 0.02
        #mass_initial, moi_initial = compute_sphere_properties(8940, radius)
        mass_initial, moi_initial = compute_sphere_properties(2710, radius)
        object_properties_approximate = ObjectProperties(
            mass=mass_initial,
            MOI=moi_initial,
            radius=initial_radius_guess,
            sigma=sigma_al,
        )
    
        # boat object
        boat_mass, boat_moi = compute_boat_properties(0.059, 0.075, 0.07, 0.031)
        #boat_mass = 0.056
        boat_properties = ObjectProperties(mass=boat_mass, MOI=boat_moi)
    
        adaptive_model = MassSysID
    
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
            if special_case_solver == 'DevinSolver':
                solver = DevinSolver(solver_model)
            else:
                solver = DevinSolver2(solver_model)
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
    
        # if inverse_dynamics:
        #     controller = PDController(env.start_state, 0.12, 0.004, 0.12, 0.004, 0, 0)
        # elif use_special_case_solver:
        #     controller = PDController(env.start_state, 0.12, 0.004, 0.12, 0.004, 0.0, 0.0)
        # else:
        #     controller = PDController(
        #         env.start_state, 0.0005, 0.000001, 0.0005, 0.000002, 0.0, 0.000
        #     )
    
        alpha = 0.9
        if special_case_solver == 'DevinSolver':
            T = 4
        elif special_case_solver == 'DevinSolver2':
            T = 2
        elif special_case_solver == 'griffin':
            T = 1
        totalmass = mass_initial + 0.056
        kp = (totalmass)/(T**2)*(alpha**2 -2*alpha + 1)
        kd = (4*(totalmass)*(1-alpha) - kp*T**2)/(2*T)
        if inverse_dynamics:
            controller = PDController(env.start_state, kp, 0.004, kd, 0.004, 0, 0)
        elif use_special_case_solver:
            controller = PDController(env.start_state, kp, 0.000, kd, 0.0, 0.0, 0.0)
        else:
            controller = PDController(
                env.start_state, kp, 0.000001, kd, 0.000002, 0.0, 0.000
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
            [controller.compute_error(transforms[0], state[0]) for state in states]
        )
    
# =============================================================================
#         # visualization stuff
#         env.radius *= 0.5
#         anim = visualization.animate_2d_path(
#             env,
#             positions,
#             transforms,
#             controls,
#             name="",
#             save_plot_steps=velocity_profile.switching_times,
#             save=False
#         )
#         npdesired_states = np.array(desired_states)
#     
#         plots = {
#             "Position (m)": np.transpose(positions),
#             "Velocity (m/s)": np.transpose(linear_velocity),
#             "Rot velocity (_/s)": np.transpose(rotational_velocity),
#             "Force ": np.transpose(np.array(wrenches)[:, :3, 0]),
#             "Torque ": np.transpose(np.array(wrenches)[:, 3:, 0]),
#             "Desired position (m)": np.transpose(npdesired_states[:, :3, 0]),
#             "Desired velocity (m/s) ": np.transpose(npdesired_states[:, 6:9, 0]),
#         }
#     
#         visualization.plot_stuff(plots, "Controller")
#     
#         plots = {
#             "Position Error(m)": np.abs(np.transpose(remaining_error[:, :3, 0])),
#             "Orientation Error(radians)": -np.transpose(remaining_error[:, 3:6, 0]),
#         }
#     
#         visualization.plot_stuff(plots, "Error")
#     
#         plots = {
#             "desired Position(m)": np.array(desired_states)[:, :3, 0].T,
#             "desired Orientation (radians)": np.array(desired_states)[:, 3:6, 0].T,
#         }
#     
#         visualization.plot_stuff(plots, "Desired Pose")
#     
#         # print('Time spent doing visualization ' + str(time.time()-visStart))
#     
#         plt.figure()
#         plt.plot(minimization_costs)
#     
#         diff = npdesired_states[:, 0, 0] - positions[:, 0]
#         plt.figure()
#         plt.plot(diff)
#         max_error = np.amax(diff)
#         rms_error = np.mean(np.square(diff))
#         
#     
#         print(np.linalg.norm(positions[-1]))
#         print(np.linalg.norm(linear_velocity[-1]))
#         print(np.linalg.norm(rotational_velocity[-1]))
#         
#         
#     
        np.savetxt(ex_filename, positions, delimiter=",")
        np.savetxt(vel_ex_filename, rotational_velocity, delimiter=",")
# =============================================================================
        import dataloader
        from datetime import datetime
    
        dataloader.save(
            "../data/sim_" + str(sim_time) + "s_" + str(datetime.now()).replace(":", "."),
            [globals()],
        )
        

