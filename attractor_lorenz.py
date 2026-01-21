# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:16:38 2025

@author: zmzhai
"""

import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
# import utils
import networkx as nx
from scipy.integrate import solve_ivp
from scipy.linalg import qr
from lyapynov import ContinuousDS, LCE
from scipy.integrate import odeint


def f_real(x, t):
    res = np.zeros_like(x)
    res[0] = 10 * (x[1] - x[0])
    res[1] = x[0] * (28 - x[2]) - x[1]
    res[2] = x[0] * x[1] - 8 / 3 * x[2]
    
    return res

def f_sindy1(x, t):
    res = np.zeros_like(x)
    res[0] = -5.311 * x[0] + 6.177 * x[1] - 0.136 * x[0] * x[2] + 0.113 * x[1] * x[2]
    res[1] = 25.167 * x[0] - 0.927 * x[0] * x[2]
    res[2] = 0.349 * x[1] - 2.679 * x[2] + 0.951 * x[0] * x[1]
    return res

def f_sindy2(x, t):
    res = np.zeros_like(x)
    res[0] = -4.023 * x[0] + 5.958 * x[1] - 0.167 * x[0] * x[2] + 0.113 * x[1] * x[2]
    res[1] = 28.361 * x[0] - 3.051 * x[1] - 1.026 * x[0] * x[2] + 0.103 * x[1] * x[2]
    res[2] = -0.105 * x[0] + 0.176 * x[1] - 2.691 * x[2] + 0.942 * x[0] * x[1]
    return res

def f_sindy3(x, t):
    res = np.zeros_like(x)
    res[0] = -9.806 * x[0] + 9.846 * x[1]
    res[1] = 24.176 * x[0] - 0.901 * x[0] * x[2]
    res[2] = -0.115 * x[0] + 0.184 * x[1] - 2.691 * x[2] + 0.942 * x[0] * x[1]
    return res

t0 = 0.
tend = 2000 
dt = 0.02
x0 = np.array([1.5, -1.5, 20.])
t_span = np.arange(t0, tend, dt)

solution_real = odeint(f_real, x0, t_span)
solution_sindy1 = odeint(f_sindy1, x0, t_span)
solution_sindy2 = odeint(f_sindy2, x0, t_span)
solution_sindy3 = odeint(f_sindy3, x0, t_span)

# Save all solutions to pickle file
solutions = {
    'real': solution_real,
    'sindy1': solution_sindy1,
    'sindy2': solution_sindy2,
    'sindy3': solution_sindy3,
    't_span': t_span,
    'x0': x0
}

# with open('./save_data/lorenz_solutions.pkl', 'wb') as f:
#     pickle.dump(solutions, f)

# fig = plt.figure(figsize=(10, 8), constrained_layout=True)
# ax = fig.add_subplot(111, projection='3d')

# ax.plot(solution_real[:, 0], solution_real[:, 1], solution_real[:, 2], label='real')
# ax.plot(solution_sindy1[:, 0], solution_sindy1[:, 1], solution_sindy1[:, 2], label='sindy1')
# ax.plot(solution_sindy2[:, 0], solution_sindy2[:, 1], solution_sindy2[:, 2], label='sindy2')
# ax.plot(solution_sindy3[:, 0], solution_sindy3[:, 1], solution_sindy3[:, 2], label='sindy3')

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.legend()

# plt.show()


import utils

kl_divergence_1 = utils.kl_divergence_attractors(solution_real[:, 0:3], solution_sindy1[:, 0:3], drop_zero=True)
kl_divergence_2 = utils.kl_divergence_attractors(solution_real[:, 0:3], solution_sindy2[:, 0:3], drop_zero=True)
kl_divergence_3 = utils.kl_divergence_attractors(solution_real[:, 0:3], solution_sindy3[:, 0:3], drop_zero=True)

# print the kl divergence
print(f"kl_divergence_1: {kl_divergence_1}")
print(f"kl_divergence_2: {kl_divergence_2}")
print(f"kl_divergence_3: {kl_divergence_3}")














































