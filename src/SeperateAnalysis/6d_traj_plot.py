# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 10:55:02 2020

@author: tabor
"""

import sys

sys.path.append("..")  # Adds higher directory to python modules path.
import dataloader
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import time
import visualization
from visualization import Lanc1,Lanc2,Lanc3

sim_time = 0

mags = 4
magnet_data_map = {2:"/home/tabor/Documents/IJRR_sim4/sim_1800s_2023-05-24 13.43.29.561066",
                   3:'/home/tabor/Documents/IJRR_sim4/sim_1800s_2023-05-24 13.48.25.239666',
                   4:'/home/tabor/Documents/IJRR_sim4/sim_1800s_2023-05-24 13.50.25.177242',
                   6:'/home/tabor/Documents/IJRR_sim4/sim_1800s_2023-05-24 13.52.37.079093',
                   -1:'/home/tabor/Documents/IJRR_sim4/sim_1800s_2023-05-24 16.29.59.054348'}


dataloader.load(
    magnet_data_map[-1], globals()
)

remaining_error_cheat = np.array(
    [controller.compute_error(desired_final_tf,state[0]) for state in states[:sim_time]]
)


dataloader.load(
    magnet_data_map[mags] , globals()
)


remaining_error = np.array(
    [controller.compute_error(desired_final_tf,state[0]) for state in states[:sim_time]]
)

if run_adaptive and not cheat:
    object_parameters_over_time = [(adaptive.parameters[0][:2].T @ np.diag((1,adaptive.problem.sigma_multiplier))).flatten()] \
                                  * (sim_time - len(adaptive.parameters))\
                                  + [(parameters[:2].T @ np.diag((1,adaptive.problem.sigma_multiplier))).flatten() for parameters in adaptive.parameters]
else:
    object_parameters_over_time = [[object_properties_approximate.radius,object_properties_approximate.sigma]] * sim_time



plt.rcParams["font.family"] = "Times New Roman"

fig, axs = plt.subplots(1, 2)
colors = [Lanc1,Lanc2,Lanc3,Lanc1,Lanc2,Lanc3]
stateNames = ["x", "y", "z", "x", "y", "z"]
plots = []
thickness = 1.0
linealpha = 1.0
dashedalpha = 0
# .set_title('position (m)')
axs[0].set_xlabel("Time (minutes)")
axs[0].set_ylabel("Position (m)")
ts = np.array(range(sim_time)) / 60.0
ts_desired = np.array(range(sim_time)) / 60.0

for j in range(2):
    for linestyle,dataset in zip(['solid','dashed'],[remaining_error,remaining_error_cheat]):
        for i in range(3):
            plots.append(
                axs[j].plot(
                    ts,
                    np.transpose(dataset[:,i + j*3, 0]),
                    color=colors[i],
                    linestyle=linestyle,
                    linewidth=thickness,
                    label=stateNames[i],
                    alpha=linealpha,
                )[0]
            )
axs[0].legend(handles=plots[:3], title="Legend", bbox_to_anchor=(1.05, 1), loc="upper left")
axs[1].set_xlabel("Time (minutes)")
axs[1].set_ylabel("Orientation (rad)")


fig.set_size_inches(7.5, 2)
plt.tight_layout()
fig_2, axis = plt.subplots()
ax2 = axis.twinx()  # instantiate a second axes that shares the same x-axis
axs_2 = (axis,ax2)

plots = []
for ((linestyle,dataset),axis) in zip(zip(['solid','dashed'],[remaining_error[:,:3,0],remaining_error[:,3:,0]]),axs_2):
    for i in range(3):
        plots.append(
            axis.plot(
                ts,
                np.transpose(dataset[:,i]),
                color=colors[i],
                linestyle=linestyle,
                linewidth=thickness,
                label=stateNames[i],
                alpha=linealpha,
            )[0]
            )

axs_2[0].legend(handles=plots[:3], title="Position", loc="lower left",ncol=3)
axs_2[0].set_xlabel("Time (minutes)")
axs_2[1].set_ylabel("Orientation (rad)") 
axs_2[0].set_ylabel("Position (m)")
axs_2[1].legend(handles=plots[3:], title="Orientation", loc="upper right",ncol=3)
axs_2[0].set_ylim([-0.06,0.06])
axs_2[1].set_ylim([-1.5, 1.5])
axs_2[0].set_xlim([0,sim_time/60])
axs_2[0].set_xlim([0,sim_time/60])
fig_2.set_size_inches(7.5,2)
plt.tight_layout()


fig_3, axis = plt.subplots()
ax2 = axis.twinx()  # instantiate a second axes that shares the same x-axis
axs_3 = (axis,ax2)

plots = []
for ((linestyle,dataset),axis) in zip(zip(['solid','dashed'],[remaining_error[:,:3,0]-remaining_error_cheat[:,:3,0],remaining_error[:,3:,0]-remaining_error_cheat[:,3:,0]]),axs_3):
    for i in range(3):
        plots.append(
            axis.plot(
                ts,
                np.transpose(dataset[:,i]),
                color=colors[i],
                linestyle=linestyle,
                linewidth=thickness,
                label=stateNames[i],
                alpha=linealpha,
            )[0]
            )


axs_3[0].legend(handles=plots[:3], title="Position", loc="lower left",ncol=3)
axs_3[0].set_xlabel("Time (minutes)")
axs_3[1].set_ylabel("Orientation (rad)") 
axs_3[0].set_ylabel("Position (m)")
axs_3[1].legend(handles=plots[3:], title="Orientation", loc="upper right",ncol=3)
axs_3[0].set_ylim([-0.002,0.002])
axs_3[1].set_ylim([-0.15, 0.15])
axs_3[0].set_xlim([0,sim_time/60])
axs_3[0].set_xlim([0,sim_time/60])
fig_3.set_size_inches(7.5,2.5)
plt.tight_layout()




folder = '/home/tabor/Documents/IJRR_sim4/' + str(mags) + "_mags"


fig.savefig(folder + "_double_fig.pdf")
fig_2.savefig(folder + "_shared_fig.pdf")
fig_3.savefig(folder + "error_shared_fig.pdf")


if mags == -1:
    env.magnets = []

env.visualization_boundary_y = ( 0.15, -0.15)
anim,anim_axis = visualization.animate_2d_path(
    env,
    positions,
    transforms,
    controls,
    object_parameters_over_time=object_parameters_over_time,
    name="",
    save_plot_steps=[1]+velocity_profile.switching_times,
    save_location = False,
)


