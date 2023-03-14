#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:40:24 2023

@author: tabor
"""


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import numpy as np


original_cm = cm.get_cmap("hot_r", 256)
newcolors = original_cm(np.concatenate([np.linspace(0.2, 0.85, 1000)]))

newcmp = ListedColormap(newcolors)

max_sigma = 8e7
min_sigma = 2e6


# https://www.wolframalpha.com/input?i=c%3D6.3*10%5E7%2F%28%281%2B%28T-20%29*3.8*10%5E-3%29%29+from+-170+to+120
# https://en.m.wikipedia.org/wiki/Electrical_resistivity_and_conductivity

important_sigmas = [
    ("Silver  \nat -35°C", 8e7),
    ("Silver", 6.3e7),
    ("Copper", 5.8e7),
    ("Gold", 4.1e7),
    ("Aluminum", 3.5e7),
    ("Platinum", 9.4e6),
    ("Titanium", 2.4e6),
]

label_sigmas = np.linspace(min_sigma, max_sigma, 9)


pixel_height = 200

width = 10


def remap(sigma):
    return pixel_height * (sigma - min_sigma) / (max_sigma - min_sigma)


x = np.linspace(0, width, width)
y = np.linspace(min_sigma, max_sigma, pixel_height)
X, Y = np.meshgrid(x, y)

fig, ax1 = plt.subplots()
ax1.set_ylim(0, pixel_height)

ax2 = ax1.twinx()

ax1.imshow(Y, cmap=newcmp, origin="lower", aspect="auto")
ax2.imshow(Y, cmap=newcmp, origin="lower", aspect="auto")
ax1.plot(np.linspace(0, pixel_height, 9))

ax1.set_xticks([])
ax2.set_xticks([])

ax1.set_yticks(remap(np.array([sigma[1] for sigma in important_sigmas])))
ax1.set_yticklabels([sigma[0] for sigma in important_sigmas])

ax2.set_yticks(remap(np.array([sigma for sigma in label_sigmas])))
ax2.set_yticklabels(["{:.1e}".format(sigma) for sigma in label_sigmas])


# ax2.set_ylim(0,pixel_height)
fig.set_figheight(4.0)
fig.set_figwidth(2.0)

fig.tight_layout()
fig.show()
fig.savefig("colormap_vertical.pdf")


fig, ax1 = plt.subplots()
ax1.set_xlim(0, pixel_height)

ax2 = ax1.twiny()

ax1.imshow(Y.T, cmap=newcmp, origin="lower", aspect="auto")
ax2.imshow(Y.T, cmap=newcmp, origin="lower", aspect="auto")
ax1.plot(np.linspace(0, width, 9))

ax1.set_yticks([])
ax2.set_yticks([])

ax2.set_xticks(remap(np.array([sigma[1] for sigma in important_sigmas])))
ax2.set_xticklabels([sigma[0] for sigma in important_sigmas], rotation=45, ha="left")

ax1.set_xticks(remap(np.array([sigma for sigma in label_sigmas])))
ax1.set_xticklabels(
    ["{:.1e}".format(sigma) for sigma in label_sigmas], rotation=-45, ha="left"
)


# ax2.set_ylim(0,pixel_height)
fig.set_figheight(1.8)
fig.set_figwidth(4.0)

fig.tight_layout()
fig.show()
fig.savefig("colormap_horizontal.pdf")
