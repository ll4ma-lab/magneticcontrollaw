# -*- coding: utf-8 -*-
"""
Created on Wed May  5 23:34:12 2021

@author: Griffin Tabor
"""
import torch
import numpy as np


def forward(x):
    clippedTheta = torch.sigmoid(x) * 720 - 360
    
    return clippedTheta


def loss_fn(x):
    error = 0.7-torch.cos(x*np.pi/180.0)
    return 1e10 * error*error


def train():
    # Construct data_loader, optimizer, etc.
    x = torch.autograd.Variable(torch.tensor(0.001), requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=5e-2)
    print(forward(x))

    for i in range(100):
        optimizer.zero_grad()
        loss_fn(forward(x)).backward()
        optimizer.step()  # This will update the shared parameters
    print(forward(x))

if __name__ == '__main__':
    import torch.multiprocessing as mp
    import time
    num_processes = 10
    # NOTE: this is required for the ``fork`` method to work
    processes = []
    start = time.time()
    train()
    print(time.time()-start)
    start = time.time()

    for rank in range(num_processes):
        p = mp.Process(target=train)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print(time.time()-start)
