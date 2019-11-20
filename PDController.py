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
        self.gains = np.zeros((halfDims,dims))

        self.gains[:quarterDims,:quarterDims] = np.eye(quarterDims) * linearKP
        self.gains[quarterDims:,quarterDims:halfDims] = np.eye(quarterDims) * rotationalKP
        self.gains[:quarterDims,halfDims:halfDims+quarterDims] = np.eye(quarterDims) * linearKD
        self.gains[quarterDims:,halfDims+quarterDims:] = np.eye(quarterDims) * rotationalKD

    
    def getDesiredWrench(self,current,desired):
            error = desired - current 
            wrench = np.matmul(self.gains,error)
            return wrench
