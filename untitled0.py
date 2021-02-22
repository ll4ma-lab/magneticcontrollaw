# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 13:26:19 2021

@author: Griffin Tabor
"""

import visualization
import matplotlib.pyplot as plt



import numpy as np
class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, r'\quad \underline{' + orig_handle + '}', usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title

plt.rcParams["font.family"] = "Times New Roman"

fig,ax = plt.subplots(1,1)
axs = [ax,0]

Lanc1 = (0,0.4470,0.7410)
Lanc2 = (0.8500,0.3250,0.0980)
Lanc3 = (0.9290,0.6940,0.1250)

colors = [Lanc1,Lanc2,Lanc3]*2
styles = ['solid','solid','solid','dashed','dashed','dashed']
stateNames = ['x','y','z','x','y','z']
plots = []
plots2 = []
thickness = 1.0
linealpha = 1.0
dashedalpha = 0
#.set_title('position (m)')
axs[0].set_xlabel('Time (minutes)')
axs[0].set_ylabel('Position (m)',)#color=colors[0])
#axs[0].tick_params(axis='y', labelcolor=colors[0])

ts = np.array(range(numSteps+1))/60.0
ts_desired = np.array(range(numSteps))/60.0 
for i in range(3):
    plots.append(axs[0].plot(ts,np.transpose(states[i:i+1,:]),color=colors[i],linestyle=styles[i],linewidth=thickness,label = stateNames[i],alpha=linealpha)[0])
    #axs[0].plot(ts_desired,npdesiredStates[:,i:i+1,0],colors[i]+'--',linewidth=thickness,alpha=dashedalpha)
#axs[0].legend(handles=plots, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.grid(True)

axs[1] = axs[0].twinx()  # instantiate a second axes that shares the same x-axis
axs[1].set_ylabel('Orientation (rad)')#,color=colors[3])
#axs[1].tick_params(axis='y', labelcolor=colors[3])

for i in range(3,6):
   plots.append(axs[1].plot(ts,np.transpose(states[i:i+1,:]),color=colors[i],label = stateNames[i],linestyle=styles[i],linewidth=thickness,alpha=linealpha)[0])
    #axs[1].plot(ts_desired,npdesiredStates[:,i:i+1,0],colors[i]+'--',linewidth=thickness,alpha=dashedalpha)
axs[1].set_ylim((-0.25,2))

p5, = ax.plot([0], marker='None',
            linestyle='None', label='dummy-tophead')

for i in range(6):
    plots.insert(i,p5)
#plots.insert(4,p5)
#plots.insert(5,p5)
categories = ['x', 'y','z']

# leg3 = ax.legend(plots,
#               ['Position'] + categories + ['Orientation'] + categories,
#               loc='upper right', ncol=2) # Two columns, vertical group labels
leg4 = ax.legend(plots,
              ['Position', '','', 'Orientation','', ''] + categories + categories,
               loc='upper right', ncol=2,frameon=False,handler_map={str: LegendTitle({'fontsize': 14})}) # Two columns, horizontal group labels
#axs[1].legend(handles=plots, title='Legend', bbox_to_anchor=(1.1, 1), loc='upper right')
fig.set_size_inches(7.5,2.5)
#plt.grid(True)

plt.tight_layout()
fig.savefig('orientation_off.pdf')