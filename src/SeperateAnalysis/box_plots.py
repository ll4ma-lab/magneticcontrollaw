# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:44:34 2021

@author: Griffin Tabor
"""
import numpy as np
import matplotlib.pyplot as plt

folder = "no_orientation"
mypath = "C:/Users/Griffin Tabor/Documents/experiments/" + folder + "/"

positionControl_orienE = np.loadtxt(mypath + "orienE.csv")
positionControl_posE = np.loadtxt(mypath + "posE.csv")
positionControl_velE = np.loadtxt(mypath + "velE.csv")
positionControl_rotvelE = np.loadtxt(mypath + "rotvelE.csv")

positionControl_orienP = np.loadtxt(mypath + "orienP.csv")
positionControl_posP = np.loadtxt(mypath + "posP.csv")
positionControl_velP = np.loadtxt(mypath + "velP.csv")
positionControl_rotvelP = np.loadtxt(mypath + "rotvelP.csv")

folder = "orientation"
mypath = "C:/Users/Griffin Tabor/Documents/experiments/" + folder + "/"

poseControl_orienE = np.loadtxt(mypath + "orienE.csv")
poseControl_posE = np.loadtxt(mypath + "posE.csv")
poseControl_velE = np.loadtxt(mypath + "velE.csv")
poseControl_rotvelE = np.loadtxt(mypath + "rotvelE.csv")

poseControl_orienP = np.loadtxt(mypath + "orienP.csv")
poseControl_posP = np.loadtxt(mypath + "posP.csv")
poseControl_velP = np.loadtxt(mypath + "velP.csv")
poseControl_rotvelP = np.loadtxt(mypath + "rotvelP.csv")

width = 0.45
ratio = 0.75
fig, axes = plt.subplots(
    1, 4, figsize=(7, 4), gridspec_kw={"width_ratios": [1, 1, ratio, ratio]}
)

axes[0].boxplot([positionControl_posE], positions=[1], widths=width)
axes[0].boxplot([poseControl_posE], positions=[2], widths=width)
axes[0].set_xticklabels(["2-DOF", "3-DOF"])
axes[0].set_xticks([1, 2])
axes[0].set_xlim(0.5, 2.5)
axes[0].title.set_text("Position \n(mm)")

axes[1].boxplot([positionControl_velE], positions=[1], widths=width)
axes[1].boxplot([poseControl_velE], positions=[2], widths=width)
axes[1].set_xticklabels(["2-DOF", "3-DOF"])
axes[1].set_xticks([1, 2])
axes[1].set_xlim(0.5, 2.5)
axes[1].title.set_text("Velocity \n(mm/s)")

axes[2].boxplot([poseControl_orienE], positions=[1], widths=width / (2 * ratio))
axes[2].set_xticklabels(["3-DOF"])
axes[2].set_xticks([1, 2])
axes[2].set_xlim(0.5, 1.5)
axes[2].title.set_text("Orientation \n(rad)")

axes[3].boxplot([poseControl_rotvelE], positions=[1], widths=width / (2 * ratio))
axes[3].set_xticklabels(["3-DOF"])
axes[3].set_xticks([1])
axes[3].set_xlim(0.5, 1.5)
axes[3].title.set_text("Angular \nVelocity (rad/s)")
fig.suptitle("Accuracy", size=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.show()


fig, axes = plt.subplots(
    1, 4, figsize=(7, 4), gridspec_kw={"width_ratios": [1, 1, 1, 1]}
)

axes[0].boxplot([positionControl_posP], positions=[1], widths=width)
axes[0].boxplot([poseControl_posP], positions=[2], widths=width)
axes[0].set_xticklabels(["2-DOF", "3-DOF"])
axes[0].set_xticks([1, 2])
axes[0].set_xlim(0.5, 2.5)
axes[0].title.set_text("Position \n(mm)")

axes[1].boxplot([positionControl_velP], positions=[1], widths=width)
axes[1].boxplot([poseControl_velP], positions=[2], widths=width)
axes[1].set_xticklabels(["2-DOF", "3-DOF"])
axes[1].set_xticks([1, 2])
axes[1].set_xlim(0.5, 2.5)
axes[1].title.set_text("Velocity \n(mm/s)")

axes[2].boxplot([poseControl_orienP], positions=[1], widths=width / (2 * ratio))
axes[2].set_xticklabels(["3-DOF"])
axes[2].set_xticks([1, 2])
axes[2].set_xlim(0.5, 1.5)
axes[2].title.set_text("Orientation \n(rad)")

axes[3].boxplot([poseControl_rotvelP], positions=[1], widths=width / (2 * ratio))
axes[3].set_xticklabels(["3-DOF"])
axes[3].set_xticks([1])
axes[3].set_xlim(0.5, 1.5)
axes[3].title.set_text("Angular \nVelocity (rad/s)")
fig.suptitle("Precision", size=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.show()
