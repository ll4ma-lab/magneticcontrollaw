# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:26:21 2021

@author: Griffin Tabor
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation


def buildSkew(vector):
    vector = np.reshape(vector, (3))
    skew = np.array(
        [
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0],
        ]
    )
    return skew


def skewToVector(skew):
    w = np.zeros((3, 1))
    w[0, 0] = skew[2, 1]
    w[1, 0] = skew[0, 2]
    w[2, 0] = skew[1, 0]
    return w


def computeError(currentTransform, desiredTransform):
    errorTransform = np.dot(np.linalg.inv(currentTransform), desiredTransform)
    linearError = errorTransform[:3, 3:]
    skew, theta = matrixLog(errorTransform[:3, :3])
    w_hat = skewToVector(skew)
    rotationError = w_hat * theta

    linearErrorW = np.dot(currentTransform[:3, :3], linearError)
    rotationErrorW = np.dot(currentTransform[:3, :3], rotationError)

    return np.concatenate((linearErrorW, rotationErrorW))


def matrixExponential(w_hat_skew, theta):
    Rot = (
        np.eye(3)
        + np.sin(theta) * w_hat_skew
        + (1 - np.cos(theta)) * np.dot(w_hat_skew, w_hat_skew)
    )
    return Rot


def matrixLog(Ae):
    phi = np.arccos(0.5 * (np.trace(Ae) - 1))
    skew = (1 / (2 * np.sin(phi))) * (Ae - np.transpose(Ae))
    return skew, phi


def sphereProperties(density, diameter):
    radius = diameter / 2.0
    volume = 4.0 / 3 * np.pi * np.power(radius, 3)
    mass = volume * density
    I = np.eye(3) * (2.0 / 5) * mass * np.power(radius, 2)
    return mass, I


dt = 1

diameter = 0.04
objectProperties = sphereProperties(8940, diameter)
state = np.zeros((12, 1))

massMatrix = np.eye(3) * objectProperties[0]
MOI = objectProperties[1]

control = np.zeros((6, 1))
from scipy.spatial.transform import Rotation as R


worldToBodyRotation = R.from_euler("xyz", state[3:6, 0])
worldToBodyR = worldToBodyRotation.as_dcm()


bodyToWorldR = np.transpose(worldToBodyR)
bodyToWorld = np.eye(4)
bodyToWorld[:3, :3] = bodyToWorldR
# bodyToWorld[:3,3] = 0.5


w = np.array([0.3, 2, 1])

theta = np.linalg.norm(w)
w_hat = w / theta
w_hat_skew = buildSkew(w_hat)

Rot = matrixExponential(w_hat_skew, theta)

Isc = np.zeros((6, 6))
Isc[:3, :3] = MOI
Isc[3:, 3:] = massMatrix
Ω0 = np.zeros((3, 1))
Ω0[:3, 0] = w[:]
V0 = np.zeros((3, 1))
V0[0, 0] = 0.25
# control[0] = 0.001

Ω = np.copy(Ω0)
V = np.copy(V0)
dt = 0.001
steps = int(1.0 / dt)
TFs = []
oldTFs = []

# PHYSICS
Ωsc = np.dot(np.transpose(bodyToWorld[:3, :3]), Ω)
Vsc = np.dot(np.transpose(bodyToWorld[:3, :3]), V)

TFs.append(np.copy(bodyToWorld))

for i in range(steps):
    rotrot = np.concatenate(
        (
            np.concatenate((np.transpose(bodyToWorld[:3, :3]), np.zeros((3, 3))), 1),
            np.concatenate((np.zeros((3, 3)), np.transpose(bodyToWorld[:3, :3])), 1),
        ),
        0,
    )
    bodyFT = np.dot(rotrot, control)

    ξsc = np.concatenate([Ωsc, Vsc])
    ΩscX = buildSkew(Ωsc)
    VscX = buildSkew(Vsc)
    ξscV = np.concatenate((np.concatenate((ΩscX, Vsc), 1), np.zeros((1, 4))), 0)

    adξ = np.concatenate(
        (np.concatenate((ΩscX, np.zeros((3, 3))), 1), np.concatenate((VscX, ΩscX), 1)),
        0,
    )
    adξs = np.transpose(adξ)

    rightSide = np.dot(adξs, np.dot(Isc, ξsc)) + bodyFT

    ξsc_dot = np.dot(np.linalg.inv(Isc), rightSide)
    bodyToStrange = np.eye(4)
    bodyToStrange[:3, :3] = bodyToWorld[:3, :3]
    bodyToWorld_dot = np.dot(bodyToStrange, ξscV)  # gsc_dot = gsc * ξscV

    bodyToWorld += bodyToWorld_dot * dt

    Ωsc += ξsc_dot[:3, :] * dt
    Vsc += ξsc_dot[3:, :] * dt

    TFs.append(np.copy(bodyToWorld))

    oldTF = np.eye(4)
    t = (i + 1) * dt
    temp = R.from_euler(
        "xyz",
        (Ω0 * t + np.transpose(control[:3] * t * t / (2 * objectProperties[1][0, 0])))[
            :, 0
        ],
    )
    oldTF[:3, :3] = temp.as_dcm()
    oldTF[:3, 3] = V0[:, 0] * t + np.transpose(
        control[3:] * t * t / (2 * objectProperties[0])
    )
    oldTFs.append(oldTF)

    if np.isnan(bodyToWorld).any():
        break
print(Rot)
print(bodyToWorld)


fig = plt.figure()
ax = p3.Axes3D(fig)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
axisLength = 0.3


def update_lines(num, TFs, lines, oldLines):
    # num = num + states.shape[1]/2

    for i in range(3):
        vector = np.zeros((4, 2))
        vector[i, 0] = axisLength
        vector[3, :] = 1
        vertices = np.dot(TFs[num], vector)
        lines[i].set_data(vertices[0:2, :])
        lines[i].set_3d_properties(vertices[2, :])
        vertices = np.dot(oldTFs[num], vector)
        oldLines[i].set_data(vertices[0:2, :])
        oldLines[i].set_3d_properties(vertices[2, :])
    return lines


lines = [
    ax.plot((0, axisLength), (0, 0), (0, 0), linewidth=2, color="r")[0],
    ax.plot((0, 0), (0, axisLength), (0, 0), linewidth=2, color="g")[0],
    ax.plot((0, 0), (0, 0), (0, axisLength), linewidth=2, color="b")[0],
]

oldLines = [
    ax.plot((0, axisLength * 2), (0, 0), (0, 0), linewidth=2, color="r", alpha=0.5)[0],
    ax.plot((0, 0), (0, axisLength), (0, 0), linewidth=2, color="g", alpha=0.5)[0],
    ax.plot((0, 0), (0, 0), (0, axisLength), linewidth=2, color="b", alpha=0.5)[0],
]

Fixedlines = [
    ax.plot((0, axisLength), (0, 0), (0, 0), linewidth=2, color="r")[0],
    ax.plot((0, 0), (0, axisLength), (0, 0), linewidth=2, color="g")[0],
    ax.plot((0, 0), (0, 0), (0, axisLength), linewidth=2, color="b")[0],
]

ax.plot(
    [TFs[j][0, 3] for j in range(i)],
    [TFs[j][1, 3] for j in range(i)],
    [TFs[j][2, 3] for j in range(i)],
)
# Creating the Animation object
line_ani = animation.FuncAnimation(
    fig, update_lines, len(TFs), fargs=(TFs, lines, oldLines), interval=1, blit=False
)

plt.show()
