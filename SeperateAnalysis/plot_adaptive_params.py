#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 23:50:18 2022

@author: tabor
"""
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import numpy as np


import dataloader
import numpy as np
import time 
import matplotlib.pyplot as plt

true_vector = [None,5.8e7]
dataloader.load("/home/tabor/experiments/double_cube/real_2022-01-28 18:40:13.245086",globals())


parameters = np.stack(adaptive.parameters)
fig,axs = plt.subplots(2,1,figsize=(5,2.5))
Lanc1 = (0,0.4470,0.7410)
Lanc2 = (0.8500,0.3250,0.0980)
Lanc3 = (0.9290,0.6940,0.1250)

i=0
axs[i].plot(parameters[:,i],c=Lanc1)
axs[i].plot([0,parameters.shape[0]],[true_vector[i],true_vector[i]],c=Lanc2)
axs[0].set_xlabel('Time (seconds)')
axs[0].set_ylabel('Radius \n(m)')
axs[0].set_xlim((0,2400))
axs[0].set_ylim((0.01,0.025))



i=1
axs[i].plot(parameters[:,i],c=Lanc1)
axs[i].plot([0,parameters.shape[0]],[true_vector[i],true_vector[i]],c=Lanc2)
axs[1].set_xlabel('Time (seconds)')
axs[1].set_ylabel('Conductivity \n(S/m)')
axs[1].set_xlim((0,2400))
axs[1].set_ylim((2.5e7,6e7))


plt.tight_layout()
plt.savefig('double_cube_params.pdf')