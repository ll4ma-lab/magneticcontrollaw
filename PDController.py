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
class PDController:
    def __init__(self,state,linearKP,rotationalKP,linearKD,rotationalKD):
        self.lastError = None
        dims = state.shape[0]
        halfDims = dims/2
        quarterDims = halfDims/2
        self.Pgains = np.zeros((halfDims,halfDims))
        self.Dgains = np.zeros((halfDims,halfDims))

        self.Pgains[:quarterDims,:quarterDims] = np.eye(quarterDims) * linearKP
        self.Pgains[quarterDims:,quarterDims:halfDims] = np.eye(quarterDims) * rotationalKP
        self.Dgains[:quarterDims,:quarterDims] = np.eye(quarterDims) * linearKD
        self.Dgains[quarterDims:,quarterDims:halfDims] = np.eye(quarterDims) * rotationalKD

    
    def getDesiredWrench(self,current,desired):
            error = desired - current 
            Pstuff = np.matmul(self.Pgains,error[:6])
            Dstuff = np.matmul(self.Dgains,error[6:])
            return Pstuff + Dstuff, Pstuff,Dstuff
