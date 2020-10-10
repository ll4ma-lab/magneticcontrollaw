# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:42:34 2020

@author: griffin
"""

import dataloader
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import visualization
import numpy as np

from os import listdir
from os.path import isfile, join
mypath = "/home/griffin/Documents/experiments/orientation/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for fileName in onlyfiles:
    print('bag' in fileName)
    if('bag' not in fileName):
        dataloader.load(mypath+fileName,globals())
        print("vel error ",totalVelSquaredError/numSteps)
        print("pos error ",totalPosSquaredError/numSteps)

#modified_controls = np.zeros((i,len(env.magnets)))
#for j in range(i):
#    modified_controls[j,controls[j][0]] = controls[j][2]
#npdesiredStates = np.array(desiredStates)
#npStates = np.array(states)