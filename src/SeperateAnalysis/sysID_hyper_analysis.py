#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 20:00:13 2022

@author: tabor
"""


start = -1000
end = -75
prob = radiusSysid(np.diag((2e5, 2e5, 0, 0, 0, 3.0)))
possible_r = np.linspace(prob.boundConstraints[0][0], prob.boundConstraints[0][1], 100)


for dif in range(5, 100, 5):
    plt.figure()
    plt.title("start " + str(start) + " dif" + str(dif))
    for start in range(-100, -900, -25):
        end = start + dif

        linearVelocity = np.array(
            [
                np.dot(state[0][:3, :3], state[2][:, 0])
                for state in states[1 + start : end - 1]
            ]
        )
        rotationalVelocity = np.array(
            [
                np.dot(state[0][:3, :3], state[1][:, 0])
                for state in states[1 + start : end - 1]
            ]
        )

        positions = np.array([state[0][:3, 3] for state in states[1 + start : end - 1]])
        some_controls = controls[start : end - 2]

        if len(some_controls) < 5:
            continue

        prob.updateVals(
            some_controls, env.magnets, positions, linearVelocity, rotationalVelocity
        )

        possibleCosts = [
            prob.cost(np.stack((possible_r[i], 1, 1)).reshape((3, 1)))
            for i in range(len(possible_r))
        ]

        plt.plot(possible_r, possibleCosts, "r")
