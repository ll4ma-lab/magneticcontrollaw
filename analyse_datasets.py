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
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

folder = 'orientation'
#folder = 'no_orientation'

mypath = "/home/tabor/Documents/experiments/"+folder + '/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

PositionErrors = []
VelocityErrors = []
orientationErrors = []
rotationalVelocityErrors = []
allStates = []


fig = plt.figure()
fig.suptitle('Paths ' + folder)

for fileName in onlyfiles:
    if('bag' not in fileName):
        dataloader.load(mypath+fileName,globals())
        #print("vel error ",totalVelSquaredError/numSteps)
        #print("pos error ",totalPosSquaredError/numSteps)

        npdesiredStates = np.array(desiredStates)
        npStates = np.array(states)
        allStates.append(npStates)
        plt.plot(npStates[:,0,0],npStates[:,1,0])

        error = npdesiredStates - npStates
        sqError =  np.power(error,2)
        absError = np.abs(error)        
        
        
        positionError = np.sqrt(sqError[:,0,0] + sqError[:,1,0])
        PositionErrors.append(positionError)
        
        
        velocityError = np.sqrt(sqError[:,6,0] + sqError[:,7,0])
        VelocityErrors.append(velocityError)
        
        orientationErrors.append(absError[:,5,0]) 
        
        
        rotationalVelocityErrors.append(absError[:,11,0])
    
    
print('average position error ' + str(np.average(PositionErrors)) +' std '+ str(np.std(PositionErrors)))
print('max position error ' + str(np.max(PositionErrors)))

print('average velocity error ' + str(np.average(VelocityErrors)) +' std '+ str(np.std(VelocityErrors)))
print('average orientation error ' + str(np.average(orientationErrors)) +' std '+ str(np.std(orientationErrors)))
print('average orientation velocity error ' + str(np.average(rotationalVelocityErrors)) +' std '+ str(np.std(rotationalVelocityErrors)))
plt.plot(npdesiredStates[:,0,0],npdesiredStates[:,1,0])

averagedStates = np.average(allStates,0)[:,:,0]
stdStates = np.std(allStates,0)
print('average(across time) of std(across trials) x ' +str(np.average(stdStates[:,0,0]))+ ' y ' +str(np.average(stdStates[:,1,0])))


stateAbove = averagedStates + 2 * stdStates[:,:,0]
stateBelow = averagedStates - 2 * stdStates[:,:,0]
plt.plot(averagedStates[:,0],averagedStates[:,1],'k--', linewidth=2)
#plt.plot(stateAbove[:,0],stateAbove[:,1],'k--', linewidth=2)
#plt.plot(stateBelow[:,0],stateBelow[:,1],'k--', linewidth=2)

splitNumb = 1
aboveSplits = np.split(stateAbove,splitNumb)
belowSplits = np.split(stateBelow,splitNumb)
averageSplits = np.split(averagedStates,splitNumb)

for i in range(splitNumb):
    
    total = np.concatenate([aboveSplits[i],np.flip(belowSplits[i],0)])
    plt.fill(total[:,0],total[:,1],'k',alpha=0.3)
    

plt.show()