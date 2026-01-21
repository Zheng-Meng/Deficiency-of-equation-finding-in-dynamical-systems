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
    return np.array([
        -2 * x[1],
        x[0] + x[2]**2,
        1 + x[1] - 2 * x[2]
    ])

def f_example1(x, t):
    return np.array([
        0.597 - 1.922 * x[1] - 0.227 * x[2],
        -0.633 + 0.970 * x[0] - 0.322 * x[2] + 0.936 * x[2]**2,
        1.018 + 0.954 * x[1] - 1.902 * x[2]
    ])

def f_example2(x, t):
    return np.array([
        0.529 - 1.991 * x[1],
        -0.458 + 0.925 * x[0] + 0.154 * x[1] - 0.618 * x[2] + 0.919 * x[2]**2,
        0.975 + 0.961 * x[1] - 1.950 * x[2]
    ])

def f_example3(x, t):
    return np.array([
        0.613 - 1.942 * x[1] - 0.182 * x[2],
        -0.588 + 0.976 * x[0] - 0.271 * x[2] + 0.939 * x[2]**2,
        0.993 + 0.955 * x[1] - 1.914 * x[2]
    ])


def f_example4(x, t):
    return np.array([
        0.655 - 1.956 * x[1] - 0.130 * x[2],
        0.883 * x[0] + 0.233 * x[1] - 0.918 * x[2] + 0.854 * x[2]**2,
        0.983 + 0.957 * x[1] - 1.933 * x[2]
    ])

def f_example5(x, t):
    return np.array([
        0.692 - 1.979 * x[1],
        -0.179 + 0.915 * x[0] + 0.194 * x[1] - 0.808 * x[2] + 0.886 * x[2]**2,
        1.020 + 0.961 * x[1] - 1.967 * x[2]
    ])

def f_example6(x, t):
    return np.array([
        0.612 - 1.912 * x[1] - 0.234 * x[2],
        -0.527 + 0.888 * x[0] + 0.198 * x[1] - 0.868 * x[2] + 0.892 * x[2]**2,
        0.985 + 0.960 * x[1] - 1.900 * x[2]
    ])



t0 = 0.
# tend = 30000
# dt = 0.1
tend = 10000
dt = 0.1
x0 = np.array([0.5, 0.5, 0.5])
t_span = np.arange(t0, tend, dt)

# Simulations
systems = {
    'real': f_real,
    'example1': f_example1,
    'example2': f_example2,
    'example3': f_example3,
    'example4': f_example4,
    'example5': f_example5,
    'example6': f_example6,
}

solutions = {'t_span': t_span, 'x0': x0}

for name, func in systems.items():
    solutions[name] = odeint(func, x0, t_span)

# with open('./save_data/sprott13_solutions.pkl', 'wb') as f:
#     pickle.dump(solutions, f)

import utils

kl_divergence_1 = utils.kl_divergence_attractors(solutions['real'][:, 0:3], solutions['example1'][:, 0:3], drop_zero=True)
kl_divergence_2 = utils.kl_divergence_attractors(solutions['real'][:, 0:3], solutions['example2'][:, 0:3], drop_zero=True)
kl_divergence_3 = utils.kl_divergence_attractors(solutions['real'][:, 0:3], solutions['example3'][:, 0:3], drop_zero=True)
kl_divergence_4 = utils.kl_divergence_attractors(solutions['real'][:, 0:3], solutions['example4'][:, 0:3], drop_zero=True)
kl_divergence_5 = utils.kl_divergence_attractors(solutions['real'][:, 0:3], solutions['example5'][:, 0:3], drop_zero=True)
kl_divergence_6 = utils.kl_divergence_attractors(solutions['real'][:, 0:3], solutions['example6'][:, 0:3], drop_zero=True)

# print the kl divergence
print(f"kl_divergence_1: {kl_divergence_1}")
print(f"kl_divergence_2: {kl_divergence_2}")
print(f"kl_divergence_3: {kl_divergence_3}")
print(f"kl_divergence_4: {kl_divergence_4}")
print(f"kl_divergence_5: {kl_divergence_5}")
print(f"kl_divergence_6: {kl_divergence_6}")
















































