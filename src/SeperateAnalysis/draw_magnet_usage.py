# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:36:02 2023

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

sphere = False
file_name = "C:/Users/tabor/Downloads/IJRR/IJRR/sphere_inv/real_2023-02-26 19_47_02.081920_adaptive_0.02_inverse_"
dataloader.load(file_name, globals())


env.radius *= 2
fig = plt.figure(figsize=(15, 15))
ax = p3.Axes3D(fig)

ax.set_xlim(*env.visualization_boundary_y)
ax.set_ylim(*env.visualization_boundary_y)
ax.set_zlim(*env.visualization_boundary_y)
# ax.set_aspect('equal')


bonus_lan_color = [123 / 255, 248 / 255, 57 / 255]
positions = np.array([state[0][:3, 3] for state in states])[-i:]


cubes = visualization.draw_cubes(env.magnets, [], ax, env.radius)
color_map = [
    visualization.Lanc1,
    visualization.Lanc2,
    visualization.Lanc3,
    bonus_lan_color,
]


for j, cube in enumerate(cubes):
    colorinfo = list(color_map[j])
    colorinfo.append(0.35)
    print(colorinfo)
    cube.set_facecolor(colorinfo)


colors = [color_map[control.mag_num] for control in controls[-i:]]

ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors)
ax.view_init(-90, -90)

ax.set_axis_off()
fig.savefig("magnet_color.pdf")
