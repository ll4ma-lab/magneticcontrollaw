#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:32:12 2022

@author: tabor
"""

from ll4ma_opt.solvers.matrix_utils import compute_inverse_cholesky

error = prob.residual(result.solution)
average_error = np.average(error)

jac = prob.jacobian(result.solution)

NM = error.shape[0]
P = prob.size()

unscaled_covariance = compute_inverse_cholesky(jac.T@jac * (1/(NM-P)))

sigma_2 = (1/(NM-P))*np.sum((error-average_error)**2)
covariance = unscaled_covariance * sigma_2