# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:02:00 2020

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
if sphere:
    dataloader.load(
        "/home/tabor/Documents/magneticcontrollaw/data/sphere_samples_4096", globals()
    )
else:
    dataloader.load(
        "/home/tabor/Documents/magneticcontrollaw/data/cylinder_samples_4096", globals()
    )


threshold = 0.1
colors = [
    (1, 0, 0, 1 - error / threshold if error < threshold else 0)
    for error in posErrorsPercent
]
print(time.time() - s)
fig = plt.figure(figsize=(8, 8))
ax = p3.Axes3D(fig)
ax.scatter(samplePoints[:, 0], samplePoints[:, 1], samplePoints[:, 2], c=colors)
ax.set_xlim(*env.visualizationBoundaryY)
ax.set_ylim(*env.visualizationBoundaryY)
ax.set_zlim(*env.visualizationBoundaryY)
visualization.draw_cubes(env.magnets, [], ax, env.radius)


ax.view_init(elev=90, azim=-90)
ax.set_zlim(-0.1, 0.1)
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(-0.1, 0.1)
plt.savefig(name + "top" + ".png")

ax.view_init(elev=49, azim=115)
plt.savefig(name + "angle" + ".png")
