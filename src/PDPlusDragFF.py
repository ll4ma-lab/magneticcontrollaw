#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:11:47 2023

@author: tabor
"""
import numpy as np
from PDController import PDController

class PDPlusDragFF(PDController):
    def get_feedforward_component(self,current,desired):
        current_tf = current[0]

        rot_vel = current_tf[:3, :3] @ current[1]
        linear_vel = current_tf[:3, :3] @ current[2]
        
        quadratic_linear_drag =  linear_vel * np.linalg.norm(linear_vel)
        quadratic_rotation_drag =  rot_vel * np.linalg.norm(rot_vel)

        external_wrench = np.concatenate((quadratic_linear_drag, quadratic_rotation_drag), 0)
        feedback_linearization = self.ff_dgains @ external_wrench
        return feedback_linearization