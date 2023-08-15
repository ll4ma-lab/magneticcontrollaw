# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:22:48 2023

@author: tabor
"""


import sys

sys.path.append("..")  # Adds higher directory to python modules path.
import dataloader
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import time
import visualization
from visualization import Lanc1, Lanc2, Lanc3

import torch


data_file = "C:/Users/tabor/OneDrive/Documents/magneticcontrollaw/data/sim_1800s_2023-06-02 15.38.14.128479"


dataloader.load(data_file, globals())
a = b
x = torch.tensor(np.transpose(positions))
xdot = torch.tensor(np.transpose(linear_velocity))
force = torch.tensor(np.transpose(np.array(wrenches)[:, :3, 0]))
desired_force = torch.tensor(np.transpose(np.array(desired_wrenchs)[:, :3, 0]))


Q = np.eye(6)

accel = force / mass


def batched_weighted_multiplication(a, Q, b):
    first = a.T @ Q
    second = torch.bmm(first.unsqueeze(1), b.unsqueeze(0).T)
    return second.squeeze(1)


time = list(range(sim_time))

top = batched_weighted_multiplication(x, Q[:3, :3], xdot)
bottom = batched_weighted_multiplication(xdot, Q[3:, 3:], accel) * 2e4
plt.plot(top, c="red")
plt.plot(bottom, c="blue")
plt.plot([0, sim_time], [0, 0], c="black")
plt.plot(top + bottom, c="green")


# ratio = top/bottom
# plt.figure()
# plt.plot(ratio)


from ll4ma_opt.problems import Problem, Constraint
from ll4ma_opt.solvers import NewtonMethod, LogBarrierSolver, AugmentedLagranMethod


class Proof(Problem):
    def __init__(self, position, velocity, acceleration):
        self._size = position.shape[0] * 2
        super().__init__()
        self.x = self.make_tensor(np.concatenate((position, velocity)))
        self.xdot = self.make_tensor(np.concatenate((velocity, acceleration)))
        self.max_bounds = np.ones((self.size(), 1)) * 100
        self.min_bounds = -np.ones((self.size(), 1)) * 100
        self.initial_solution = np.random.random((self.size(), 1))

    def tensor_cost(self, x):
        V = x.reshape((self._size, self._size))
        Q = V.T @ V
        discrete_time_stability = batched_weighted_multiplication(self.x, Q, self.xdot)
        costs = torch.sigmoid(discrete_time_stability) - 0.5
        cost = torch.sum(costs)
        return cost

    def size(self):
        return self._size**2


class StableConstraint(Constraint):
    def __init__(self, x, xdot):
        self.x = x
        self.xdot = xdot

        self._size = x.shape[0]

        super().__init__()

    def tensor_error(self, x):
        V = x.reshape((self._size, self._size))
        Q = V.T @ V
        error = self.x.T @ Q @ self.xdot
        return -error + 1


class ProofConstrained(Problem):
    def __init__(self, position, velocity, acceleration):
        self._size = position.shape[0] * 2
        super().__init__()
        self.x = self.make_tensor(np.concatenate((position, velocity)))
        self.xdot = self.make_tensor(np.concatenate((velocity, acceleration)))
        self.max_bounds = np.ones((self.size(), 1)) * 100
        self.min_bounds = -np.ones((self.size(), 1)) * 100
        self.inequality_constraints = [
            StableConstraint(self.x[:, i], self.xdot[:, i])
            for i in range(self.x.shape[1])
        ]
        self.initial_solution = np.random.random((self.size(), 1))

    def tensor_cost(self, x):
        cost = 0.0 * x.T @ x
        return cost

    def size(self):
        return self._size**2


start = 1200
end = 1400

params = {"alpha": 1, "rho": 0.9, "min_alpha": 1e-30}


prob = ProofConstrained(x[0:1, start:end], xdot[0:1, start:end], accel[0:1, start:end])
prob = Proof(x[0:1, start:end], xdot[0:1, start:end], accel[0:1, start:end])

optimizer = NewtonMethod(prob, alpha=100)
# optimizer = LogBarrierSolver(prob, "newton", params, init_barrier_multiplier=.01, update_ratio=10.1, use_phase_I_sum=False)
# optimizer = AugmentedLagranMethod(prob, "newton", params,print_info=True)


result = optimizer.optimize(prob.initial_solution, max_iterations=1000)
result.display()
Q = result.solution.reshape((prob._size, prob._size))
Q = torch.tensor(np.diag([1, 1e5]))
test = batched_weighted_multiplication(prob.x, Q, prob.xdot)
plt.figure()
plt.plot(test)
plt.figure()
[plt.plot(result.iterates[:, i, 0]) for i in range(prob.size())]
