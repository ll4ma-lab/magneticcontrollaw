# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:50:44 2020

@author: tabor
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import visualization
from IPython.display import HTML


fig = plt.figure()

ax = fig.gca(projection="3d")

ax.set_xlim(*[-1,1])
ax.set_ylim(*[-1,1])
ax.set_zlim(*[-1,1])


ax.set_title('old Axis')
x_axis, = ax.plot([-1,0,1], [0,0,0], [0,0,0], lw=2,color='red')
y_axis, = ax.plot([0,0,0], [-1,0,1], [0,0,0], lw=2,color='green')
z_axis, = ax.plot([0,0,0], [0,0,0], [-1,0,1], lw=2,color='blue')
cube_definition=visualization.buildRectangularPrism([0,0,0],0.1,0.1,0.1)
visualization.plot_cube(cube_definition,[1,0,0,1],ax)
visualization.drawSphere([1,0,0],0.2,ax)
plt.show()


fig = plt.figure()

ax = fig.gca(projection="3d")

ax.set_xlim(*[-1,1])
ax.set_ylim(*[-1,1])
ax.set_zlim(*[-1,1])


ax.set_title('new Axis')
x_axis, = ax.plot([-1,0,1], [0,0,0], [0,0,0], lw=2,color='red')
line, = ax.plot([], [], [], lw=2,color='green')
visualization.plot_cube(cube_definition,[1,0,0,1],ax)
visualization.drawSphere([1,0,0],0.2,ax)
def animate(i):
    x = [0,0]
    y = [0,np.sin(np.deg2rad(i*4))]
    z = [0,np.cos(np.deg2rad(i*4))]
    line.set_data(x, y)
    line.set_3d_properties(z)
    fig.canvas.draw()
    return line
anim = animation.FuncAnimation(fig, animate, frames=90, interval=10)
anim.save('new_axis.gif', writer='imagemagick', fps=30)
plt.show()
