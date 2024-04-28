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
from visualization import Lanc1, Lanc2, Lanc3

sim_time = 0
input_folder = "/home/tabor/Downloads"
input_folder = "C:/Users/tabor/OneDrive/Documents/testing"

mags = 2
magnet_data_map = {
    2: input_folder + "/sim_900s_2024-01-26 22.33.17.873123",
    3: input_folder + "/sim_900s_2024-01-26 22.35.17.747402",
    4: input_folder + "/IJRR_sim4/sim_1800s_2023-05-24 13.50.25.177242",
    6: input_folder + "/IJRR_sim4/sim_1800s_2023-05-24 13.52.37.079093",
    -1: input_folder + "/sim_900s_2024-01-26 22.33.08.055640",
}


dataloader.load(magnet_data_map[-1], globals())
states_cheat = states[:sim_time]
remaining_error_cheat = np.array(
    [
        controller.compute_error(desired_final_tf, state[0])
        for state in states[:sim_time]
    ]
)


true_file = magnet_data_map[mags]
#true_file = "C:/Users/tabor/OneDrive/Documents/magneticcontrollaw/data/sim_1800s_2024-01-26 16.26.30.829454"
#true_file = "C:/Users/tabor/OneDrive/Documents/magneticcontrollaw/data/sim_1800s_2024-01-26 16.26.29.800482"
dataloader.load(true_file, globals())


remaining_error = np.array(
    [
        controller.compute_error(desired_final_tf, state[0])
        for state in states[:sim_time]
    ]
)

comparison_error = np.array(
    [
        controller.compute_error(cheat_state[0], state[0])
        for cheat_state, state in zip(states_cheat[:sim_time], states[:sim_time])
    ]
)


if run_adaptive and not cheat:
    object_parameters_over_time = [
        (
            adaptive.parameters[0][:2].T
            @ np.diag((1, adaptive.problem.sigma_multiplier))
        ).flatten()
    ] * (sim_time - len(adaptive.parameters)) + [
        (parameters[:2].T @ np.diag((1, adaptive.problem.sigma_multiplier))).flatten()
        for parameters in adaptive.parameters
    ]
else:
    object_parameters_over_time = [
        [object_properties_approximate.radius, object_properties_approximate.sigma]
    ] * sim_time


plt.rcParams["font.family"] = "Times New Roman"
ts = np.array(range(sim_time)) / 60.0
ts_desired = np.array(range(sim_time)) / 60.0
colors = [Lanc1, Lanc2, Lanc3, Lanc1, Lanc2, Lanc3]
stateNames = ["x", "y", "z", "x", "y", "z"]
thickness = 1.0
linealpha = 1.0
dashedalpha = 0


def double_plot(axs):
    plots = []

    # .set_title('position (m)')
    axs[0].set_xlabel("Time (minutes)")
    axs[0].set_ylabel("Position (m)")

    for j in range(2):
        for linestyle, dataset in zip(
            ["solid", "dashed"], [remaining_error, remaining_error_cheat]
        ):
            for i in range(3):
                plots.append(
                    axs[j].plot(
                        ts,
                        np.transpose(dataset[:, i + j * 3, 0]),
                        color=colors[i],
                        linestyle=linestyle,
                        linewidth=thickness,
                        label=stateNames[i],
                        alpha=linealpha,
                    )[0]
                )
    axs[0].legend(
        handles=plots[:3], title="Legend", bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    axs[1].set_xlabel("Time (minutes)")
    axs[1].set_ylabel("Orientation (rad)")


def shared_plot(axis):
    ax2 = axis.twinx()  # instantiate a second axes that shares the same x-axis
    axs_2 = (axis, ax2)

    plots = []
    for (linestyle, dataset), axis in zip(
        zip(
            ["solid", "dashed"], [remaining_error[:, :3, 0], remaining_error[:, 3:, 0]]
        ),
        axs_2,
    ):
        for i in range(3):
            plots.append(
                axis.plot(
                    ts,
                    np.transpose(dataset[:, i]),
                    color=colors[i],
                    linestyle=linestyle,
                    linewidth=thickness,
                    label=stateNames[i],
                    alpha=linealpha,
                )[0]
            )

    axs_2[0].legend(handles=plots[:3], title="Position", loc="lower left", ncol=3)
    axs_2[0].set_xlabel("Time (minutes)")
    axs_2[1].set_ylabel("Orientation (rad)")
    axs_2[0].set_ylabel("Position (m)")
    axs_2[1].legend(handles=plots[3:], title="Orientation", loc="upper right", ncol=3)
    axs_2[0].set_ylim([-0.06, 0.06])
    axs_2[1].set_ylim([-1.5, 1.5])
    axs_2[0].set_xlim([0, sim_time / 60])
    axs_2[0].set_xlim([0, sim_time / 60])


def error_plot(axis):
    plots = []
    ax2 = axis.twinx()  # instantiate a second axes that shares the same x-axis
    axs_3 = (axis, ax2)
    for (linestyle, dataset), axis in zip(
        zip(
            ["solid", "dashed"],
            [comparison_error[:, :3, 0], comparison_error[:, 3:, 0]],
        ),
        axs_3,
    ):
        for i in range(3):
            plots.append(
                axis.plot(
                    ts,
                    np.transpose(dataset[:, i]),
                    color=colors[i],
                    linestyle=linestyle,
                    linewidth=thickness,
                    label=stateNames[i],
                    alpha=linealpha,
                )[0]
            )

    axs_3[0].legend(handles=plots[:3], title="Position", loc="lower left", ncol=3)
    axs_3[0].set_xlabel("Time (minutes)")
    axs_3[1].set_ylabel("Orientation (rad)")
    axs_3[0].set_ylabel("Position (m)")
    axs_3[1].legend(handles=plots[3:], title="Orientation", loc="upper right", ncol=3)
    axs_3[0].set_ylim([-0.002, 0.002])
    axs_3[1].set_ylim([-0.15, 0.15])
    axs_3[0].set_xlim([0, sim_time / 60])
    axs_3[0].set_xlim([0, sim_time / 60])


def l2_data_plot(axis, dataset, lims):
    ax2 = axis.twinx()  # instantiate a second axes that shares the same x-axis
    axs_2 = (axis, ax2)
    thickness = 1.5
    plots = []
    l2_colors = [Lanc1, Lanc3]
    l2_names = ["Position (m)", "Orientation (rad)"]
    for i in range(2):
        plots.append(
            axs_2[i].plot(
                ts,
                np.linalg.norm(dataset[:, (i * 3) : (i * 3 + 3), 0], axis=1),
                color=l2_colors[i],
                linewidth=thickness,
                label=l2_names[i],
                alpha=linealpha,
            )[0]
        )
        axs_2[i].yaxis.label.set_color(l2_colors[i])
        axs_2[i].tick_params(axis="y", colors=l2_colors[i])
        # axs_2[i].spines['left'].set_color(l2_colors[i])
        axs_2[i].set_ylim(lims[i])

    axs_2[0].legend(handles=plots[:3], loc="upper right", ncol=1)
    axs_2[0].set_xlabel("Time (minutes)")
    # axs_2[1].set_ylabel("Orientation (rad)")
    # axs_2[0].set_ylabel("Position (m)")
    # axs_2[1].legend(handles=plots[3:], title="Orientation", loc="upper right",ncol=3)
    # axs_2[0].set_ylim([-0.06,0.06])
    # axs_2[1].set_ylim([-1.5, 1.5])
    axs_2[0].set_xlim([0, sim_time / 60])
    axs_2[0].set_xlim([0, sim_time / 60])
    axs_2[0].spines["left"].set_edgecolor(l2_colors[0])  # y axis line
    axs_2[1].spines["right"].set_edgecolor(l2_colors[1])  # y axis line

    axs_2[1].spines["left"].set_edgecolor(l2_colors[0])  # y axis line
    axs_2[0].spines["right"].set_edgecolor(l2_colors[1])  # y axis line

    axs_2[0].spines["right"].set_linewidth(thickness)  # y axis line
    axs_2[0].spines["left"].set_linewidth(thickness)  # y axis line
    axs_2[0].spines["top"].set_linewidth(thickness)  # y axis line
    axs_2[0].spines["bottom"].set_linewidth(thickness)  # y axis line


fig, axs = plt.subplots(1, 2)
double_plot(axs)
fig.set_size_inches(7.5, 2)
plt.tight_layout()


fig_2, axis = plt.subplots()
shared_plot(axis)
fig_2.set_size_inches(7.5, 2)
plt.tight_layout()


fig_3, axis = plt.subplots()
error_plot(axis)
fig_3.set_size_inches(7.5, 2.5)
plt.tight_layout()

fig_4, axis = plt.subplots()
l2_data_plot(axis, remaining_error, [[-0.01, 0.08], [-0.2, 1.6]])
fig_4.set_size_inches(5, 2.5)
plt.tight_layout()

folder = input_folder + "/IJRR_sim4/" + str(mags) + "_mags"


fig.savefig(folder + "_double_fig.pdf")
fig_2.savefig(folder + "_shared_fig.pdf")
fig_3.savefig(folder + "error_shared_fig.pdf")


if mags == -1:
    env.magnets = []

env.visualization_boundary_y = (0.15, -0.15)
# anim, anim_fig = visualization.animate_2d_path(
#     env,
#     positions,
#     transforms,
#     controls,
#     object_parameters_over_time=object_parameters_over_time,
#     name="",
#     save_plot_steps=[1] + velocity_profile.switching_times,
#     save_location=folder,
#     mag_color=[164 / 256.0, 159 / 256.0, 12 / 256.0, 0.2],
# )


fig = plt.figure(figsize=plt.figaspect(2.0))
# fig.suptitle()

# First subplot
ax = fig.add_subplot(1, 2, 1)
l2_data_plot(ax, remaining_error, [[-0.0, 0.08], [-0, 2.0]])
ax.title.set_text("Pose Error")

ax = fig.add_subplot(1, 2, 2)
l2_data_plot(ax, comparison_error, [[-0.000, 0.004], [-0.0, 0.1]])
ax.title.set_text("Trajectory Error")

fig.set_size_inches(6.5, 3)

plt.tight_layout()
fig.savefig(folder + "l2_double.pdf")
