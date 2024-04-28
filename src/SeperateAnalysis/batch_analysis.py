# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:48:01 2024

@author: tabor
"""

import sys
sys.path.append("..")  # Adds higher directory to python modules path.
import matplotlib.pyplot as plt
import numpy as np
from visualization import Lanc1, Lanc2, Lanc3


import dataloader

num_plots = 2

if num_plots == 3:
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
else:
    fig, (ax2,ax3) = plt.subplots(1, 2)

#ax1 = fig.add_subplot(1,1,1)
#ax2 = fig.add_subplot(2,1,1)

file_names = ['C:/Users/tabor/OneDrive/Documents/magneticcontrollaw/data/batch_CylinderControlCommand_4magnetd_s_2024-02-08 05.26.14.455676',
              'C:/Users/tabor/OneDrive/Documents/magneticcontrollaw/data/batch_SphereControlCommand_4magnetd_s_2024-02-08 04.33.31.589790']


file_names = ['C:/Users/tabor/OneDrive/Documents/magneticcontrollaw/data/batch_CylinderControlCommand_4magnetd_s_2024-02-08 23.33.34.495404',
              'C:/Users/tabor/OneDrive/Documents/magneticcontrollaw/data/batch_SphereControlCommand_4magnetd_s_2024-02-08 22.33.35.357162']


file_names = ['C:/Users/tabor/OneDrive/Documents/magneticcontrollaw/data/batch_CylinderControlCommand_2magnetd_s_2024-03-25 23.07.18.582485',
              'C:/Users/tabor/OneDrive/Documents/magneticcontrollaw/data/batch_SphereControlCommand_2magnetd_s_2024-03-25 19.28.19.901399']



position_medians = []
orientation_medians = []

plot_names = ['Prior','New']

colors = {'Nature':Lanc1,'Ours':Lanc3}

for index,(file_name,plot_name) in enumerate(zip(file_names,plot_names)):
    dataloader.load(file_name, globals())

    
    
    c = (0,0,0)
    median_color = 'r'


    np_data = np.array(convergence_data).T
    np_data[1:] = np.abs(np_data[1:]/0.99)

    if num_plots == 3:
        ax1.boxplot(np_data[0],positions=[index],labels=[plot_name])
        ax1.title.set_text('time')

    fill_alpha = 0.0


    for ax,data in zip([ax2,ax3],np_data[1:]):
        ax.boxplot(data,positions=[index],
                labels=[plot_name],
                notch=True,
                patch_artist=True,
                boxprops=dict(facecolor=list(c) + [fill_alpha], color=c),
                capprops=dict(color=c),
                whiskerprops=dict(color=c),
                flierprops=dict(color=c, markeredgecolor=c),
                medianprops=dict(color=median_color),
                sym='+',
                widths=0.8,
                showfliers=False)

    position_medians.append(np.median(np_data[1]))
    orientation_medians.append(np.median(np_data[2]))
    print(np.max(np_data[0]))
ax2.title.set_text('Translation')
ax3.title.set_text('Rotation')

ax2.set_ylim(1,2.6)
ax3.set_ylim(1,2.6)

#ax2.tick_params(axis='x', labelrotation=45)
#ax3.tick_params(axis='x', labelrotation=45)

#fig.suptitle('Regularization Path Length')
#ax3.set_yticks(range(1,8))
fig.set_size_inches(3,3.4)
plt.tight_layout()
plt.savefig('regular_2mag.pdf')


improvement = position_medians[0]/position_medians[1]
print(improvement,1/improvement)
improvement = orientation_medians[0]/orientation_medians[1]
print(improvement,1/improvement)


improvement = (position_medians[0]-1)/(position_medians[1]-1)
print(improvement,1/improvement)
improvement = (orientation_medians[0]-1)/(orientation_medians[1]-1)
print(improvement,1/improvement)
