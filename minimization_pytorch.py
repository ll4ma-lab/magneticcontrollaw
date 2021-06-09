# -*- coding: utf-8 -*-
"""
Created on Wed May  5 23:34:12 2021

@author: Griffin Tabor
"""
import torch
import numpy as np

@torch.jit.script
def forward(x):
    clippedTheta = torch.sigmoid(x) * 720 - 360
    
    return clippedTheta

@torch.jit.script
def loss_fn(x):
    error = 0.7-torch.cos(x*np.pi/180.0)
    return 1e10 * error*error



def train(_=0):
    x = torch.autograd.Variable(torch.tensor(0.001), requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=5e-2)

    for i in range(100):
        optimizer.zero_grad()
        loss_fn(forward(x)).backward()
        optimizer.step()  # This will update the shared parameters
    return forward(x).detach()

if __name__ == '__main__':
    import torch.multiprocessing as mp
    import time
    num_processes = 10
    # NOTE: this is required for the ``fork`` method to work
    processes = []
    start = time.time()
    train()
    print('single run time ',time.time()-start)

    start = time.time()
    for i in range(10): 
        train()
    print('single threaded time ',time.time()-start)


    probs = [0] * 10
    start = time.time()
    with mp.Pool(processes=4) as pool:
        print(time.time()-start)

        middle = time.time()
        a=pool.map(train, probs)
        print('first time ',time.time()-middle)
        
        middle = time.time()
        a=pool.map(train, probs)
        print('second time ',time.time()-middle)
        
    print(time.time()-start)
