# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:24:22 2019

@author: tabor
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:57:35 2019

@author: tabor
"""

import numpy as np
from se3Utils import buildSkew, buildTF, matrixLog, skewToVector


class PDController:
    def __init__(
        self,
        state,
        linearKP,
        rotationalKP,
        linearKD,
        rotationalKD,
        linearFFD=0.0,
        rotationalFFD=0.0,
    ):
        self.lastError = None
        dims = state.shape[0]
        halfDims = int(dims / 2)
        quarterDims = int(halfDims / 2)
        self.Pgains = np.zeros((halfDims, halfDims))
        self.Dgains = np.zeros((halfDims, halfDims))
        self.FFDgains = np.zeros((halfDims, halfDims))

        self.Pgains[:quarterDims, :quarterDims] = np.eye(quarterDims) * linearKP
        self.Pgains[quarterDims:, quarterDims:halfDims] = (
            np.eye(quarterDims) * rotationalKP
        )

        self.Dgains[:quarterDims, :quarterDims] = np.eye(quarterDims) * linearKD
        self.Dgains[quarterDims:, quarterDims:halfDims] = (
            np.eye(quarterDims) * rotationalKD
        )

        self.FFDgains[:quarterDims, :quarterDims] = np.eye(quarterDims) * linearFFD
        self.FFDgains[quarterDims:, quarterDims:halfDims] = (
            np.eye(quarterDims) * rotationalFFD
        )

        self.lastDesired = None

    def computeError(self, currentTransform, desiredTransform):
        errorTransform = np.dot(np.linalg.inv(currentTransform), desiredTransform)
        linearError = errorTransform[:3, 3:]
        skew, theta = matrixLog(errorTransform[:3, :3])
        if theta == 0.0:
            rotationError = np.zeros((3, 1))
        else:
            w_hat = skewToVector(skew)
            rotationError = w_hat * theta

        linearErrorW = np.dot(currentTransform[:3, :3], linearError)
        rotationErrorW = np.dot(currentTransform[:3, :3], rotationError)
        return np.concatenate((linearErrorW, rotationErrorW))

    def getDesiredWrench(self, current, desired):

        currentTF = current[0]
        desiredTF = buildTF(desired[:3, 0], desired[3:6, 0])

        positionError = self.computeError(currentTF, desiredTF)
        # (bodyToWorld,Ωsc,Vsc)

        rotVelError = desired[6:9] - np.dot(currentTF[:3, :3], current[1])
        linearVelError = desired[6:9] - np.dot(currentTF[:3, :3], current[2])
        velError = np.concatenate((linearVelError, rotVelError), 0)

        Pstuff = np.matmul(self.Pgains, positionError)
        Dstuff = np.matmul(self.Dgains, velError)

        total = Pstuff + Dstuff
        if self.lastDesired is not None:
            ffError = desired - self.lastDesired
            FFstuff = np.matmul(self.FFDgains, ffError[6:])
            total += FFstuff

        self.lastDesired = np.copy(desired)
        return total, Pstuff, Dstuff
