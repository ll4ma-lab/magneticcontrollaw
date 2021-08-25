#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:01:16 2021

@author: tabor
"""

import numpy as np
import matplotlib.pyplot as plt

force = np.genfromtxt('forceSpacesysID.csv',delimiter=',')
accel = np.genfromtxt('accelSpacesysID.csv',delimiter=',')

fig, (ax1, ax2) = plt.subplots(1, 2,figsize =(3, 3))
ax1.plot(force[0],np.log(force[1]))
ax2.plot(accel[0],np.log(accel[1]))
ax1.yaxis.set_visible(False)
ax2.yaxis.set_visible(False)

ax1.set_title('force space')
ax2.set_title('accel space')

ax1.set_xlabel('radius (m)')
ax2.set_xlabel('radius (m)')

fig.tight_layout()