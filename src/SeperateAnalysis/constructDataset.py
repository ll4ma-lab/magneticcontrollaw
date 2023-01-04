# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:36:41 2020

@author: tabor
"""
#del cylinderModel, cylinderSolver,model
from cylinderModel import cylinderSolver,cylinderModel
model = cylinderModel(env.magnets,diameter)
solver = cylinderSolver(env.magnets)
solver.setup(model,env.trajectoryDimensions,objectProperties,env.tolerances)

lin = 0.000000002 * numSteps
rot = 0.0000000005 * numSteps
from scipy.interpolate import UnivariateSpline
ts = list(range(numSteps))
x_spline = UnivariateSpline(ts, npStates[:,0],s=lin)
xdot_spline = x_spline.derivative()
xdotdot_spline = xdot_spline.derivative()

y_spline = UnivariateSpline(ts, npStates[:,1],s=lin)
ydot_spline = y_spline.derivative()
ydotdot_spline = ydot_spline.derivative()

theta_spline = UnivariateSpline(ts, npStates[:,5],s=rot)
thetadot_spline = theta_spline.derivative()
thetadotdot_spline = thetadot_spline.derivative()

fig,axs = plt.subplots(3)
fig.suptitle('Smoothed Data')
xs = np.linspace(np.min(ts),np.max(ts), 1000)
nameList = ['x','y','theta']
splines = [[x_spline,xdot_spline,xdotdot_spline],[y_spline,ydot_spline,ydotdot_spline],[theta_spline,thetadot_spline,thetadotdot_spline]]
data = [npStates[:,0],npStates[:,1],npStates[:,5]]
for index in range(len(nameList)):

    axs[index].set_title(nameList[index])
    axs[index].plot(ts, data[index], 'ro', ms=5)

    axs[index].plot(xs, splines[index][0](xs), 'g', lw=3)
    axs[index].plot(xs, splines[index][1](xs)*100, 'b', lw=3)         
    axs[index].plot(xs, splines[index][2](xs)*100*100, 'y', lw=3)         

fig.canvas.draw()





accel = np.transpose(np.array([[xdotdot_spline(ts[i]) for i in range(numSteps)],
                               [ydotdot_spline(ts[i]) for i in range(numSteps)],[0 for i in range(numSteps)],[0 for i in range(numSteps)],[0 for i in range(numSteps)],
                               [thetadotdot_spline(ts[i]) for i in range(numSteps)]]))

        
accelFD = npStates[:-1,6:,0] - npStates[1:,6:,0] 
observedForces = accel[:,:3]*objectProperties[0]
observedTorques = np.matmul(accel[:,3:],objectProperties[1])

desiredForces = np.array([pComponents[i] + dComponents[i] for i in range(numSteps)])[:,:,0]

expectedForces = []
for i in range(numSteps):
    expectedForces.append(model.forwardPass(states[i],controls[i]))
npExpectedForces = np.array(expectedForces)

i = 500
dummy_control = solver.solveWrench(model,states[i],expectedForces[i],env.trajectoryDimensions)
print(dummy_control)


fig,axs = plt.subplots(2)
colors = ['c','m','y','c','m','y']
stateNames = ['x','y','z','x','y','z']
plots = []
thickness = 1.0
linealpha = 1.0
dashedalpha = 1
#.set_title('position (m)')
axs[0].set_xlabel('Time (minutes)')
axs[0].set_ylabel('Position (m)')
ts = np.array(range(numSteps+1))/60.0
ts_desired = np.array(range(numSteps))/60.0 
for i in range(2):
    plots.append(axs[0].plot(observedForces[:,i],colors[i],linewidth=thickness,label = stateNames[i],alpha=linealpha)[0])
    axs[0].plot(npExpectedForces[:,i],colors[i]+'--',linewidth=thickness,alpha=dashedalpha)
    #axs[0].plot(desiredForces[:,i],colors[i]+'+',linewidth=thickness,alpha=dashedalpha)

axs[0].legend(handles=plots, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
axs[1].set_xlabel('Time (minutes)')
axs[1].set_ylabel('Orientation (rad)')
for i in range(2,3):
    plots.append(axs[1].plot(observedTorques[:,i],colors[i],linewidth=thickness,label = stateNames[i],alpha=linealpha)[0])
    axs[1].plot(npExpectedForces[:,3+i],colors[i]+'--',linewidth=thickness,alpha=dashedalpha)