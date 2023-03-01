# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:40:03 2021

@author: Griffin Tabor
"""
import numpy as np
import time
from se3Utils import build_skew, repair_rot_matrix
from tqdm import tqdm
from solvers.modelOptimization import SphereControlCommand
from ObjectProperties import build_mass_matrix
from se3Utils import build_tf


def state_noise(state):

    transform = np.copy(state[0])

    current_linear_velocity = transform[:3, :3] @ state[2]
    current_rotational_velocity = transform[:3, :3] @ state[1]

    position_noise = 0  # np.random.normal(0,3e-4,(3))
    linear_velocity_noise = 0  # np.random.normal(0,3e-5,(3,1))
    rotational_velocity_noise = 0  # np.random.normal(0,1e-5,(3,1))

    transform[:3, 3] += position_noise
    current_linear_velocity += linear_velocity_noise
    current_rotational_velocity += rotational_velocity_noise

    tf_inverse = np.linalg.inv(transform)[:3, :3]

    state = [
        transform,
        tf_inverse @ current_rotational_velocity,
        tf_inverse @ current_linear_velocity,
    ]
    return state


def simulate_experiment(
    simulator,
    controller,
    env,
    velocity_profile,
    model,
    solver,
    adaptive,
    sim_time=10,
    cheat=False,
    debug_prints=True,
    run_adaptive=False,
    inverse_dynamics=False,
):
    state = simulator.state
    states = []
    wrenches = []
    desired_wrenchs = []
    desired_states = []
    controls = []
    start = time.time()
    minimization_costs = []
    times = []

    iterates = range(1, sim_time + 1)
    if debug_prints:
        iterates = tqdm(iterates)

    for i in iterates:
        # if i == 300 and runAdaptive:
        #    model.cylinderModel.object_properties.radius *=1.5
        # arbitrarily change radius without updating mass
        desired_state = velocity_profile.get_desired_state(
            simulator.simulator_time, state
        )
        desired_wrench, p, d = controller.get_desired_wrench(state, desired_state)

        desired_states.append(desired_state)
        desired_wrenchs.append(desired_wrench)
        if cheat:

            if inverse_dynamics:
                inverted_accel = np.zeros(desired_wrench.shape)
                inverted_accel[:3] = desired_wrench[3:]
                inverted_accel[3:] = desired_wrench[:3]
                inverted_wrench = simulator.isc @ inverted_accel

                true_wrench = np.zeros(inverted_wrench.shape)
                true_wrench[:3] = inverted_wrench[3:]
                true_wrench[3:] = inverted_wrench[:3]
            else:
                true_wrench = desired_wrench
                control = SphereControlCommand()

        else:
            control = solver.solve_wrench(state[0][:3, 3], desired_wrench)
            true_wrench = model.forward_pass(state[0][:3, 3], control)
            #true_wrench[3:]=0
            wrench_error = true_wrench - desired_wrench
            cost = np.sum((wrench_error[:3, :] ** 2)) + 200 * np.sum(
                (wrench_error[3:, :] ** 2)
            )
            minimization_costs.append(cost)
        wrenches.append(true_wrench)
        controls.append(control)
        times.append(simulator.simulator_time)

        env.problem_specific_modification(state, true_wrench)
        if run_adaptive:
            adaptive.step(state_noise(state), control)
        states.append(state)
        state = simulator.simulate(true_wrench, control.seconds)

    if debug_prints:
        run_time = time.time() - start
        print(run_time)

    return (
        states,
        wrenches,
        desired_wrenchs,
        desired_states,
        controls,
        minimization_costs,
    )


class DiscreteTimeSimulator:
    def __init__(self, state, object_properties):
        self.dt = 0.002

        body_to_world = build_tf(state[:3, 0], state[3:6, 0])

        world_to_body_r = body_to_world[:3, :3].T

        ωsc = world_to_body_r @ state[9:12, :]
        vsc = world_to_body_r @ state[6:9, :]
        self.state = (body_to_world, ωsc, vsc)

        self.isc = build_mass_matrix(object_properties)

        self.simulator_time = 1e-1
        self.time = 0

    def simulate(self, control, seconds):
        self.state = self.forward_pass(self.state, control, seconds)
        return self.state

    def simulator_step(self, body_to_world, ωsc, vsc, FT, dt):
        rotrot = np.concatenate(
            (
                np.concatenate(
                    (np.transpose(body_to_world[:3, :3]), np.zeros((3, 3))), 1
                ),
                np.concatenate(
                    (np.zeros((3, 3)), np.transpose(body_to_world[:3, :3])), 1
                ),
            ),
            0,
        )
        TF = np.zeros(FT.shape)
        TF[:3] = FT[3:]
        TF[3:] = FT[:3]
        body_tf = np.dot(rotrot, TF)

        ξsc = np.concatenate([ωsc, vsc])
        ωsc_x = build_skew(ωsc)
        vsc_x = build_skew(vsc)
        ξsc_v = np.concatenate((np.concatenate((ωsc_x, vsc), 1), np.zeros((1, 4))), 0)

        adξ = np.concatenate(
            (
                np.concatenate((ωsc_x, np.zeros((3, 3))), 1),
                np.concatenate((vsc_x, ωsc_x), 1),
            ),
            0,
        )
        adξs = np.transpose(adξ)

        right_side = np.dot(adξs, np.dot(self.isc, ξsc)) + body_tf

        ξsc_dot = np.dot(np.linalg.inv(self.isc), right_side)
        body_to_strange = np.eye(4)
        body_to_strange[:3, :3] = body_to_world[:3, :3]
        body_to_world_dot = np.dot(body_to_strange, ξsc_v)  # gsc_dot = gsc * ξscV

        body_to_world += body_to_world_dot * dt
        body_to_world = repair_rot_matrix(body_to_world)

        ωsc += ξsc_dot[:3, :] * dt
        vsc += ξsc_dot[3:, :] * dt
        return body_to_world, ωsc, vsc

    def forward_pass(self, state, FT, seconds):
        steps = int(seconds / self.dt)
        start = time.time()

        # Notation and math from https://downloads.hindawi.com/journals/ijae/2017/4034328.pdf
        (body_to_world0, ωsc0, vsc0) = state

        body_to_world = np.copy(body_to_world0)
        ωsc = np.copy(ωsc0)
        vsc = np.copy(vsc0)

        FT = np.reshape(FT, (6, 1))

        for i in range(steps):
            body_to_world, ωsc, vsc = self.simulator_step(
                body_to_world, ωsc, vsc, FT, self.dt
            )

        time_left = seconds - self.dt * steps
        body_to_world, ωsc, vsc = self.simulator_step(
            body_to_world, ωsc, vsc, FT, time_left
        )
        self.simulator_time += seconds

        self.time += time.time() - start

        return (body_to_world, ωsc, vsc)


# def setState(self,state):
#     self.state = np.copy(state)
