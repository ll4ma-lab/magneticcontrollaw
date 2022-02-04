# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:42:34 2020

@author: griffin
"""

import sys
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append("../scripts") # Adds higher directory to python modules path.
import visualization
import numpy as np
import dataloader

from environments import PlanarProblem,SpaceProblem

np.set_printoptions(threshold=np.inf)
#np.set_printoptions(precision=3)

import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
from matplotlib.patches import Ellipse

folder = 'nature'
folder = 'double_cube'
folder='dummy'
folder = 'base_real'

#folder = 'no_orientation'

mypath = "/home/tabor/experiments/" + folder + "/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

PositionErrors = []
VelocityErrors = []
orientationErrors = []
rotationalVelocityErrors = []
allStates = []

Lanc1 = (0,0.4470,0.7410)
Lanc2 = (0.8500,0.3250,0.0980)
Lanc3 = (0.9290,0.6940,0.1250)
fig = plt.figure()
name = 'Paths ' + folder
#fig.suptitle(name)

for fileName in onlyfiles:
    if('bag' not in fileName and 'pdf' not in fileName and 'csv' not in fileName and 'dir' not in fileName and 'bak' not in fileName):
        if folder == 'nature':
            dataloader.load(mypath+fileName,globals())
        else:
            dataloader.load((mypath+fileName)[:-4],globals())

        print(fileName)
        if 'useSphere' in globals():
            from se3Utils import buildSkew,buildTF,matrixLog,skewToVector
            errors = []
            numSteps = 2000

            for k in range(numSteps):
                desired = desiredStates[k+(len(states)-numSteps)]
                current =  states[k+(len(states)-numSteps)]
                currentTF = current[0]
                desiredTF = buildTF(desired[:3,0],desired[3:6,0])
                positionError = controller.computeError(currentTF, desiredTF)
                
                rotVelError = desired[6:9]-np.dot(currentTF[:3,:3],current[1])
                linearVelError = desired[6:9]-np.dot(currentTF[:3,:3],current[2])
                velError = np.concatenate((linearVelError,rotVelError),0)
                errors.append(np.concatenate((positionError,velError)))
            error = np.array(errors)
        else:

            npdesiredStates = np.array(desiredStates)
            npStates = np.array(states)
            npStates[:,1,:] *=-1 #flip y to make plots z down
            npdesiredStates[:,1,:] *=-1
            
            allStates.append(npStates)
            plt.plot(npStates[:,0,0],npStates[:,1,0],'k',alpha=0.5)
    
            error = npdesiredStates - npStates
            
        
        sqError =  np.power(error,2)
        absError = np.abs(error)        
        
        
        positionError = np.sqrt(sqError[:,0,0] + sqError[:,1,0])

        PositionErrors.append(positionError)
        
        
        velocityError = np.sqrt(sqError[:,6,0] + sqError[:,7,0])
        VelocityErrors.append(velocityError)
        
        orientationErrors.append(absError[:,5,0]) 


        
        rotationalVelocityErrors.append(absError[:,11,0])
    

print('average position error ' + str(np.average(PositionErrors)) +' std '+ str(np.std(PositionErrors)) + ' max ' + str(np.max(np.abs(PositionErrors))))

print('average velocity error ' + str(np.average(VelocityErrors)) +' std '+ str(np.std(VelocityErrors)) + ' max ' + str(np.max(np.abs(VelocityErrors))))
print('average orientation error ' + str(np.average(orientationErrors)) +' std '+ str(np.std(orientationErrors)) + ' max ' + str(np.max(np.abs(orientationErrors))))
print('average orientation velocity error ' + str(np.average(rotationalVelocityErrors)) +' std '+ str(np.std(rotationalVelocityErrors)) + ' max ' + str(np.max(np.abs(rotationalVelocityErrors))))

averagedStates = np.average(allStates,0)[:,:,0]
stdStates = np.std(allStates,0)
print('average(across time) of std(across trials) x ' +str(np.average(stdStates[:,0,0]))+ ' y ' +str(np.average(stdStates[:,1,0])))

plt.plot(npdesiredStates[240:,0,0],npdesiredStates[240:,1,0],color=Lanc1,linewidth=4)
fig.axes[0].tick_params(axis='both', which='major', labelsize=20)
fig.axes[0].tick_params(axis='both', which='minor', labelsize=20)

num = 5
width = 0.1

plt.xticks(np.linspace(-width,width,num))
plt.yticks(np.linspace(-width,width,num))


stateAbove = averagedStates + 2 * stdStates[:,:,0]
stateBelow = averagedStates - 2 * stdStates[:,:,0]
#plt.plot(averagedStates[:,0],averagedStates[:,1],'y--', linewidth=2)
#plt.plot(stateAbove[:,0],stateAbove[:,1],'k--', linewidth=2)
#plt.plot(stateBelow[:,0],stateBelow[:,1],'k--', linewidth=2)

npAllStates = np.array(allStates)
npAllStates[:,:,[0,1,6,7],0]*=1000

print(npAllStates.shape)
npAllCov = np.zeros((6,6,numSteps))

npPrecisionValues =np.zeros((4,numSteps))

multiplier = 5.991
ellipseDataset = []
for i in range(numSteps):
    
    
    properData = np.transpose(npAllStates[:,i,[0,1,5,6,7,11],0])
    npAllCov[:,:,i] = np.cov(properData)
    
    
    detPartialCov = np.linalg.det(np.cov(properData[:2,:]))
    detPartialCovVel = np.linalg.det(np.cov(properData[3:5,:]))

    npPrecisionValues[0,i] =  np.sqrt(detPartialCov)
    npPrecisionValues[1,i] =  np.std(properData[2,:])
    npPrecisionValues[2,i] =  np.sqrt(detPartialCovVel)
    npPrecisionValues[3,i] =  np.std(properData[5,:])


    eigval,eigvec = np.linalg.eig(npAllCov[:2,:2,i])
    
    maxIndex = np.argmax(abs(eigval)) #cant remember if eigvals are always positive
    angle = np.rad2deg(np.arctan2(eigvec[1,maxIndex],eigvec[0,maxIndex]))
    width = np.sqrt(multiplier*eigval[maxIndex])*2/1000 #large axis   * numberSTDs * 2(radius to diamter) * m/mm
    height = np.sqrt(multiplier*eigval[maxIndex-1])*2/1000 #small axis   * numberSTDs * 2(radius to diamter) * m/mm
    #print(width,stdStates[i,0,0]*multiplier*2)
    ells = Ellipse(xy=averagedStates[i,:2],
                width=width, height=height,#2x to get to radius from diameter 
                angle=angle)
    if i > 240:
        desired = [npdesiredStates[i,0,0],npdesiredStates[i,1,0]]
    else: 
        desired = [0,0]
        
    trials_data = np.concatenate([[npAllStates[k,i,0,0],npAllStates[k,i,1,0]] for k in range(5)]).tolist()
    #print(trials_data)
    ellipseDatum = [averagedStates[i,0],averagedStates[i,1], desired[0],desired[1],width,height,angle]
    ellipseDataset.append(trials_data+ellipseDatum[:7])
    fig.axes[0].add_artist(ells)
    ells.set_alpha(1)
    ells.set_facecolor(Lanc3)
    ells.set_clip_box(fig.axes[0].bbox)
fig.set_size_inches(8,8)
fig.axes[0].set_aspect('equal')
print('showing')
plt.show()
fig.savefig(name + '.pdf')
header = []
for i in range(5):
    header.append('x_trial_'+str(i))
    header.append('y_trial_'+str(i))
header = header + ['x_average','y_average','x_desired','y_desired','ellipse_width','elipse_height','elipse_rotation']
print('setup header')
npellipseDataset = np.array(ellipseDataset)
print(npellipseDataset.shape)
np.savetxt(mypath+'/ellipse_dataset.csv',npellipseDataset,delimiter=',',header = ','.join(header))
print('saved csv')
cov = np.average(npAllCov,2)
#np.savetxt(folder+'_cov.csv',cov,delimiter=',')
print(cov)
precision = np.average(npPrecisionValues,1)
print('precision averages ',precision)
print('std of generalized std ' + str(np.std(npPrecisionValues[0,:])))
print('std of std of rotation ' + str(np.std(npPrecisionValues[1,:])))
print('std of generalized velocity std ' + str(np.std(npPrecisionValues[2,:])))
print('std of std of rotation velocity ' + str(np.std(npPrecisionValues[3,:])))






def boxPlot(data,dataname,constant=1):
    fig2 = plt.figure(figsize =(1.5, 4))
    data = abs(np.hstack(data))*constant
    plt.boxplot(data)
    fig2.axes[0].tick_params(axis='both', which='major', labelsize=12)
    fig2.axes[0].tick_params(axis='both', which='minor', labelsize=12)
    plt.tight_layout()
    #plt.show()
    name = mypath + "/" + dataname
    np.savetxt(name + '.csv',data,delimiter=',')

   #print(name)
    fig2.savefig(name + '.pdf')



boxPlot(PositionErrors,'posE',1000)
boxPlot(VelocityErrors,'velE',1000)
boxPlot(orientationErrors,'orienE')
boxPlot(rotationalVelocityErrors,'rotvelE')

boxPlot(npPrecisionValues[0,:],'posP')
boxPlot(npPrecisionValues[2,:],'velP')
boxPlot(npPrecisionValues[1,:],'orienP')
boxPlot(npPrecisionValues[3,:],'rotvelP')

#boxPlot(VelocityErrors,'velE')
#boxPlot(orientationErrors,'orienE')
#boxPlot(rotationalVelocityErrors,'rotvelE')