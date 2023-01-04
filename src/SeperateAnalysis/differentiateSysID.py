#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 19:54:41 2022

@author: tabor
"""

prob = radiusSysid(
    np.diag((2e3, 2e3, 0, 0, 0, 0.0)), physical=True
)  # built with M(0.04) Q M(0.04)
prob.updateVals(
    some_controls, env.magnets, positions, noisylinearVelocity, noisyrotationalVelocity
)


prob2 = radiusSysid(
    np.diag((0, 0, 0, 0, 0, 1e0)), physical=True
)  # built with M(0.04) Q M(0.04)
prob2.updateVals(
    some_controls, env.magnets, positions, noisylinearVelocity, noisyrotationalVelocity
)

possible_r = np.linspace(
    prob.boundConstraints[0][0], prob.boundConstraints[0][1] * 0.5, 100
)

mult = prob.initial_solution[1][0]
mult2 = prob.initial_solution[2][0]
possibleCosts = [
    prob.cost(np.stack((possible_r[i], mult, mult2)).reshape((3, 1)))
    for i in range(len(possible_r))
]
possibleCosts2 = [
    prob2.cost(np.stack((possible_r[i], mult, mult2)).reshape((3, 1)))
    for i in range(len(possible_r))
]

fig = plt.figure()
plt.plot(possible_r, possibleCosts, "r")
plt.plot(possible_r, possibleCosts2, "b")
plt.plot(possible_r, np.array(possibleCosts) + np.array(possibleCosts2), "g")
