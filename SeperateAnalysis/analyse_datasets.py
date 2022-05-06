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
from matplotlib.patches import Polygon

def rms(error,axis=None):
    return np.sqrt(np.mean(error **2,axis=axis))



#folder = 'nature'
#folder = 'double_cube'
#folder = 'base_real'
#folder= 'U'
#folder = 'cuboid'
#folder = 'copper_sphere'
#folder = 'no_orientation'
folder = 'cylinder'
#folder = 'aluminum_sphere'
#folder = 'copper_sphere_lower_lower_bounds'
#folder = 'copper_half_conductivity'
#folder = 'copper_double_conductivity'
#folder = 'copper_half_both'
#folder = 'copper_larger_radius'
#folder = 'copper_double_both'
#folder = 'square'
#folder = 'different_radius'
#mypath = "/media/tabor/Storage/experiments/" + folder + "/"
mypath = '/media/tabor/Storage/new_experiments/' + folder + '/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

PositionErrors = []
VelocityErrors = []
orientationErrors = []
rotationalVelocityErrors = []
allStates = []

Lanc1 = (0,0.4470,0.7410)
Lanc2 = (0.8500,0.3250,0.0980)
Lanc3 = (0.9290,0.6940,0.1250)
UtahU = (204/256,0,0)

fig = plt.figure()
fig.add_subplot()
all_adaptive_params = []

parameter_fig,parameter_ax = plt.subplots((2),sharex=True,gridspec_kw={'height_ratios': [3,1]})

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
            #numSteps = 2000
            positions = []
            for k in range(numSteps):
                desired = desiredStates[k+(len(states)-numSteps)]
                current =  states[k+(len(states)-numSteps)]
                currentTF = current[0]
                positions.append(currentTF[:3,3])
                desiredTF = buildTF(desired[:3,0],desired[3:6,0])
                positionError = controller.computeError(currentTF, desiredTF)
                rotVelError = desired[6:9]-np.dot(currentTF[:3,:3],current[1])
                linearVelError = desired[6:9]-np.dot(currentTF[:3,:3],current[2])
                velError = np.concatenate((linearVelError,rotVelError),0)
                errors.append(np.concatenate((positionError,velError)))
            error = np.array(errors)
            
            npStates = np.array(positions)
            npdesiredStates = np.array(desiredStates)
            
            npStates[:,1] *=-1 #flip y to make plots z down
            npdesiredStates[:,1] *=-1
            allStates.append(npStates.T)

            fig.axes[0].plot(npStates[:,0],npStates[:,1],c=Lanc1,alpha=0.5)
            
            if runAdaptive:
                #first parameter was held until 1/2 batch size(25)
                adaptive_parameters = np.array([adaptive.parameters[0]] * 12 + adaptive.parameters)
                all_adaptive_params.append(adaptive_parameters)
                #parameter_ax[0].plot(adaptive_parameters[:,0],'k',alpha=0.6)
                #parameter_ax[1].plot(adaptive_parameters[:,1],c=Lanc1,alpha=0.6)
                print(adaptive_parameters.shape)
        else:

            npdesiredStates = np.array(desiredStates)
            npStates = np.array(states)
            npStates[:,1,:] *=-1 #flip y to make plots z down
            npdesiredStates[:,1,:] *=-1
            
            allStates.append(npStates)
            fig.axes[0].plot(npStates[:,0,0],npStates[:,1,0],'k',alpha=0.5)
    
            error = npdesiredStates - npStates
            
        
        sqError =  np.power(error,2)
        absError = np.abs(error)        
        
        
        positionError = np.sqrt(sqError[:,0,0] + sqError[:,1,0])

        PositionErrors.append(positionError)
        
        
        velocityError = np.sqrt(sqError[:,6,0] + sqError[:,7,0])
        VelocityErrors.append(velocityError)
        
        orientationErrors.append(absError[:,5,0]) 


        
        rotationalVelocityErrors.append(absError[:,11,0])

adaptive_plot_max_t = 600

#parameter_ax[1].plot([0,adaptive_plot_max_t],[3.77E+07,3.77E+07],'k--')
#parameter_fig.suptitle('Adapted parameters over time')
parameter_ax[0].set_ylabel('Radius (m)')
parameter_ax[1].set_ylabel('Conductivity')


if runAdaptive:
    np_adaptive_params = np.array(all_adaptive_params)
    average_params = np.mean(np_adaptive_params,0)
    std_params = np.std(np_adaptive_params,0)

    
    i=0
    parameter_ax[0].plot(all_adaptive_params[i][:adaptive_plot_max_t,0],c=Lanc3)
    parameter_ax[1].plot(all_adaptive_params[i][:adaptive_plot_max_t,1],c=Lanc3)

    i=1
    parameter_ax[0].plot(all_adaptive_params[i][:adaptive_plot_max_t,0],c=Lanc2)
    parameter_ax[1].plot(all_adaptive_params[i][:adaptive_plot_max_t,1],c=Lanc2)


    #parameter_ax[0].plot(average_params[:adaptive_plot_max_t,0],c=Lanc1)
    #parameter_ax[1].plot(average_params[:adaptive_plot_max_t,1],c=Lanc1)

    deviations = 1
    upper_bound = (average_params+std_params*deviations)[:adaptive_plot_max_t,:,0]
    lower_bound = (average_params-std_params*deviations)[:adaptive_plot_max_t,:,0]
    
    bounds = np.concatenate((upper_bound[:,0],np.flip(lower_bound[:,0])))
    timesteps = list(range(adaptive_plot_max_t))
    
    poly_points = np.stack((np.concatenate((timesteps,np.flip(timesteps))),bounds)).T
    #poly_points = np.stack((timesteps,upper_bound[:,0])).T
    poly = Polygon(poly_points,facecolor = Lanc2)
    #parameter_ax[0].add_patch(poly)

    #parameter_ax[0].plot(upper_bound[:,0],c=UtahU)
    #parameter_ax[0].plot(lower_bound[:,0],c=UtahU)

    #parameter_ax[0].set_xticks(np.linspace(0,adaptive_plot_max_t,11))
    parameter_ax[1].set_xticks(np.linspace(0,adaptive_plot_max_t,6))
    
parameter_ax[0].plot([0,adaptive_plot_max_t],[0.02,0.02],'k--')
parameter_ax[1].plot([0,adaptive_plot_max_t],[5.8E+07,5.8E+07],'k--')
    
#parameter_ax[0].set_yticks(np.linspace(0.01,0.025,4))
parameter_ax[0].set_yticks(np.linspace(0.01,0.04,7))

parameter_ax[1].set_yticks(np.linspace(2e7,7e7,3))
parameter_fig.set_size_inches(5,3)
parameter_fig.tight_layout()



print('average position error ' + str(np.average(PositionErrors)) +' std '+ str(np.std(PositionErrors)) + ' max ' + str(np.max(np.abs(PositionErrors))))
print('average velocity error ' + str(np.average(VelocityErrors)) +' std '+ str(np.std(VelocityErrors)) + ' max ' + str(np.max(np.abs(VelocityErrors))))
print('average orientation error ' + str(np.average(orientationErrors)) +' std '+ str(np.std(orientationErrors)) + ' max ' + str(np.max(np.abs(orientationErrors))))
print('average orientation velocity error ' + str(np.average(rotationalVelocityErrors)) +' std '+ str(np.std(rotationalVelocityErrors)) + ' max ' + str(np.max(np.abs(rotationalVelocityErrors))))

print('RMS position '+str(rms(np.array(PositionErrors)))+' RMS orientation ' + str(rms(np.array(orientationErrors))))
print('RMS position '+str([rms(PositionErrors[i]) for i in range(len(PositionErrors))])+' RMS orientation ' + str([rms(orientationErrors[i]) for i in range(len(PositionErrors))]))

print('Average position error '+str([np.average(PositionErrors[i]) for i in range(len(PositionErrors))])+' Average orientation ' + str([np.average(orientationErrors[i]) for i in range(len(PositionErrors))]))


#fig.axes[0].plot(npdesiredStates[240:,0,0],npdesiredStates[240:,1,0],color=Lanc1,linewidth=4)
fig.axes[0].tick_params(axis='both', which='major', labelsize=20)
fig.axes[0].tick_params(axis='both', which='minor', labelsize=20)


    
npAllStates = np.array(allStates).squeeze()
if 'useSphere' in globals():
    npAllStates = np.transpose(npAllStates,(0,2,1))
averagedStates = np.average(npAllStates,0)

npAllStates[:,:,[0,1]]*=1000

print(npAllStates.shape)
npAllCov = np.zeros((6,6,numSteps))

npPrecisionValues =np.zeros((4,numSteps))

multiplier = 5.991
ellipseDataset = []
for i in range(numSteps):
    
    
    properData = np.transpose(npAllStates[:,i,[0,1]])
    npAllCov[:2,:2,i] = np.cov(properData)
    
    
    detPartialCov = np.linalg.det(np.cov(properData[:2,:]))
    detPartialCovVel = np.linalg.det(np.cov(properData[3:5,:]))

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
        
    trials_data = np.concatenate([[npAllStates[k,i,0],npAllStates[k,i,1]] for k in range(npAllStates.shape[0])]).tolist()
    #print(trials_data)
    ellipseDatum = [averagedStates[i,0],averagedStates[i,1], desired[0],desired[1],width,height,angle]
    ellipseDataset.append(trials_data+ellipseDatum[:7])
    fig.axes[0].add_artist(ells)
    ells.set_alpha(1)
    ells.set_facecolor(Lanc2)
    ells.set_clip_box(fig.axes[0].bbox)
    
num = 5
width = 0.1

fig.axes[0].set_xticks(np.linspace(-width,width,num))
fig.axes[0].set_yticks(np.linspace(-width,width,num))


    
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
# #np.savetxt(folder+'_cov.csv',cov,delimiter=',')
# print(cov)
# precision = np.average(npPrecisionValues,1)
