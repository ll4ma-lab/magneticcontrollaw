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


force = np.genfromtxt("forceSpacesysID.csv", delimiter=",")
accel = np.genfromtxt("accelSpacesysID.csv", delimiter=",")

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
ax1.plot(force[0], np.log(force[1]), label="Force", color=Lanc3)
ax1.plot(
    accel[0],
    np.log(accel[1]) + np.log(force[1, 959] / accel[1, 959]),
    label="Accel",
    linestyle="dashed",
    color=Lanc1,
)


ax1.set_xlabel("Radius (m)")
ax1.set_ylabel("Log Loss")
ax1.legend()

ax1.xaxis.set_ticks((0, 0.02, 0.04, 0.06, 0.08, 0.1))

fig.tight_layout()
