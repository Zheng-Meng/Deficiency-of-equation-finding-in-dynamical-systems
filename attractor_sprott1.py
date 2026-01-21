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
        x[1] * x[2],
        x[0] - x[1],
        1 - x[0] * x[1]
    ])


def f_example1(x, t):
    return np.array([
        -0.164 * x[1] + 0.953 * x[1] * x[2],
        0.969 * x[0] - 0.972 * x[1],
        0.871 - 0.864 * x[0] * x[1]
    ])

def f_example2(x, t):
    return np.array([
        0.123 * x[2] + 1.023 * x[1] * x[2],
        1.024 * x[0] - 1.013 * x[1],
        0.776 - 0.237 * x[1] - 0.912 * x[0] * x[1]
    ])


def f_example3(x, t):
    return np.array([
        0.977 * x[1] * x[2],
        -0.173 + 1.009 * x[0] - 1.016 * x[1],
        0.851 - 0.917 * x[0] * x[1]  # 2313.101 - 975.424 - 1336.826 = 0.851
    ])


def f_example4(x, t):
    return np.array([
        0.103 * x[0] - 0.262 * x[1] + 0.921 * x[1] * x[2],
        0.957 * x[0] - 1.013 * x[1],
        0.905 - 0.911 * x[0] * x[1]  # 1635.500 - 1634.595 = 0.905
    ])

def f_example5(x, t):
    return np.array([
        -0.227 * x[1] - 0.127 * x[2] + 0.856 * x[1] * x[2],
        -0.38 + 0.946 * x[0] - 0.961 * x[1],
        0.958 + 0.609 * x[1] - 0.886 * x[0] * x[1]
    ])


def f_example6(x, t):
    return np.array([
        -0.158 * x[1] + 0.954 * x[1] * x[2],
        -0.141 + 0.964 * x[0] - 0.964 * x[1],
        0.792 - 0.864 * x[0] * x[1]  # 3393.071 - 1810.866 - 1581.413 = 0.792
    ])


t0 = 0.
# tend = 30000
# dt = 0.1
tend = 10000
dt = 0.1
x0 = np.array([0.1, 0.1, 0.1])
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

# with open('./save_data/sprott1_solutions.pkl', 'wb') as f:
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
















































