#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:01:16 2021

@author: tabor
"""

import numpy as np
import matplotlib.pyplot as plt

Lanc3 = (0, 0.4470, 0.7410)
Lanc1 = (0.8500, 0.3250, 0.0980)
Lanc2 = (0.9290, 0.6940, 0.1250)


import sys

sys.path.append("..")  # Adds higher directory to python modules path.


from sysIDoptimization import RadiusSysid
from solvers.modelOptimization import build_acceleration_q, build_force_q
from extract_sys_id import extract_sys_id_data


file_name = "C:/Users/tabor/OneDrive/Documents/magneticcontrollaw/data/sim_1800s_2023-06-13 19.17.57.945537"
# file_name = 'C:/Users/tabor/OneDrive/Documents/magneticcontrollaw/data/sim_1620s_2023-06-13 19.36.01.181783'
# file_name = 'C:/Users/tabor/OneDrive/Documents/magneticcontrollaw/data/sim_5400s_2023-06-13 20.02.51.638405'
# file_name = "C://Users//tabor//Downloads//IJRR/IJRR//known_sphere//real_2023-03-07 15_11_55.936480_MultiMagSolver"


physical = False


(
    linear_velocity,
    rotational_velocity,
    positions,
    selected_controls,
    initial_object_properties,
    boat,
    env,
) = extract_sys_id_data(
    file_name,
    alt_rotational_velocity=False,
    skip_end=0,
    skip_beginning=0,
)


if physical:
    drag_terms = (8.12e-3, 2.15, 1.13e-5, 1.90e-4)
    dimensions = "xyt"
else:
    dimensions = "all"
    boat = False
    drag_terms = (0, 0, 0, 0)

# boat = False
sysID_type = RadiusSysid

prob = sysID_type(
    build_acceleration_q(dimensions),
    object_properties=initial_object_properties,
    boat=boat,
    drag_terms=drag_terms,
)  # built with M(0.04) Q M(0.04)
prob.update_vals(
    selected_controls,
    env.magnets,
    positions,
    linear_velocity,
    rotational_velocity,
)
prob.regularization_multiplier = 0
num_points = 1000
possible_r = np.linspace(prob.min_bounds[0], prob.max_bounds[0], num_points)[:, 0]

acceleration_costs = np.array(
    [
        prob.cost(
            np.stack((possible_r[i], prob.initial_solution[1, 0])).reshape((2, 1))
        )
        for i in range(len(possible_r))
    ]
)

prob = sysID_type(
    build_force_q(dimensions),
    object_properties=initial_object_properties,
    boat=boat,
    force_mode=True,
    drag_terms=drag_terms,
)  # built with M(0.04) Q M(0.04)
prob.update_vals(
    selected_controls,
    env.magnets,
    positions,
    linear_velocity,
    rotational_velocity,
)
prob.regularization_multiplier = 0

wrench_costs = np.array(
    [
        prob.cost(
            np.stack((possible_r[i], prob.initial_solution[1, 0])).reshape((2, 1))
        )
        for i in range(len(possible_r))
    ]
)


true_value_index = np.argmin((possible_r - 0.02) ** 2)
shift = acceleration_costs[true_value_index] / wrench_costs[true_value_index]
acceleration_costs /= shift

force = (possible_r, wrench_costs)
accel = (possible_r, acceleration_costs)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3, 3))
ax1.plot(force[0], np.log(force[1]))
ax2.plot(accel[0], np.log(accel[1]))
ax1.yaxis.set_visible(False)
ax2.yaxis.set_visible(False)

ax1.set_title("force space")
ax2.set_title("accel space")

ax1.set_xlabel("radius (m)")
ax2.set_xlabel("radius (m)")

ax1.xaxis.set_ticks((0, 0.02, 0.1))

fig.tight_layout()


fig, ax1 = plt.subplots(1, 1, figsize=(3, 3))
ax2 = ax1.twinx()
ax1.plot(force[0], np.log(force[1]))
ax2.plot(accel[0], np.log(accel[1]))
ax1.yaxis.set_visible(False)
ax2.yaxis.set_visible(False)

ax1.set_title("force space")
ax2.set_title("accel space")

ax1.set_xlabel("radius (m)")
ax2.set_xlabel("radius (m)")

ax1.xaxis.set_ticks((0, 0.02, 0.1))

fig.tight_layout()


fig, ax1 = plt.subplots(1, 1, figsize=(3, 2))
ax1.plot(force[0], np.log(force[1]), label="Wrench", color=Lanc3)
ax1.plot(
    accel[0],
    np.log(accel[1]),
    label="Accel",
    linestyle="dashed",
    color=Lanc1,
)


ax1.set_xlabel("Radius (m)")
ax1.set_ylabel("Log Loss")
ax1.legend()

ax1.xaxis.set_ticks((0, 0.02, 0.04, 0.06, 0.08, 0.1))

fig.tight_layout()
