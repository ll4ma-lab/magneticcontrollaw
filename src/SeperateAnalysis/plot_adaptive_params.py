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

experiment = 'alum'


if experiment == 'double_cube':
    true_vector = [None,5.8e7]
    dataloader.load("/home/tabor/experiments/double_cube/real_2022-01-28 18:40:13.245086",globals())
    fig_title = 'double_cube_params'
elif experiment == 'copper':
    true_vector = [0.02,5.8e7]
    dataloader.load("/home/tabor/experiments/0.02_copper_adaptive/real_2022-01-28 14:04:22.953992",globals())
    fig_title = 'copper_params'
elif experiment == 'alum':
    true_vector = [0.02,3.77E+07]
    dataloader.load("/home/tabor/experiments/0.02_aluminum/real_2022-01-28 16:38:04.960314",globals())
    fig_title = 'alum_params'

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
axs[1].set_ylim((2.5e7,7e7))


plt.tight_layout()
plt.savefig(fig_title+'.pdf')