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


def f_example1(x, t):
    return np.array([
        -9.815 * x[0] + 9.322 * x[1] + 0.107 * x[2],
        26.331 * x[0] - 0.237 * x[1] - 0.958 * x[0] * x[2],
        0.444 * x[1] - 2.701 * x[2] + 0.956 * x[0] * x[1]
    ])

def f_example2(x, t):
    return np.array([
        -9.731 * x[0] + 9.609 * x[1],
        31.072 * x[0] - 3.569 * x[1] - 1.092 * x[0] * x[2] + 0.103 * x[1] * x[2],
        -0.104 * x[0] + 0.344 * x[1] - 2.703 * x[2] + 0.947 * x[0] * x[1]
    ])

def f_example3(x, t):
    return np.array([
        -4.933 * x[0] + 6.115 * x[1] - 0.145 * x[0] * x[2] + 0.114 * x[1] * x[2],
        29.493 * x[0] - 3.173 * x[1] - 1.062 * x[0] * x[2] + 0.103 * x[1] * x[2],
        0.262 * x[1] - 2.670 * x[2] + 0.943 * x[0] * x[1]
    ])

def f_example4(x, t):
    return np.array([
        -5.654 * x[0] + 6.624 * x[1] - 0.128 * x[0] * x[2] + 0.105 * x[1] * x[2],
        25.078 * x[0] - 0.930 * x[0] * x[2],
        0.293 * x[1] - 2.681 * x[2] + 0.942 * x[0] * x[1]
    ])

def f_example5(x, t):
    return np.array([
        -9.806 * x[0] + 9.846 * x[1],
        24.176 * x[0] - 0.901 * x[0] * x[2],
        -0.113 * x[0] + 0.184 * x[1] - 2.691 * x[2] + 0.942 * x[0] * x[1]
    ])

def f_example6(x, t):
    return np.array([
        -1.939 * x[0] + 5.172 * x[1] - 0.215 * x[0] * x[2] + 0.124 * x[1] * x[2],
        22.199 * x[0] + 0.388 * x[1] - 0.846 * x[0] * x[2],
        -2.720 * x[2] + 0.945 * x[0] * x[1]
    ])


# additional for 30% missing data
def f_example7(x, t):
    return np.array([
        -9.836 * x[0] + 9.692 * x[1],
        24.836 * x[0] - 0.921 * x[0] * x[2],
        0.278 * x[1] - 2.708 * x[2] + 0.947 * x[0] * x[1]
    ])

def f_example8(x, t):
    return np.array([
        -6.287 * x[0] + 6.849 * x[1] - 0.109 * x[0] * x[2] + 0.101 * x[1] * x[2],
        25.355 * x[0] - 0.937 * x[0] * x[2],
        -0.148 * x[0] + 0.365 * x[1] - 2.614 * x[2] + 0.950 * x[0] * x[1]
    ])

def f_example9(x, t):
    return np.array([
        -9.741 * x[0] + 9.745 * x[1],
        30.713 * x[0] - 3.710 * x[1] - 1.083 * x[0] * x[2] + 0.109 * x[1] * x[2],
        -0.310 * x[0] + 0.275 * x[1] - 2.719 * x[2] + 0.944 * x[0] * x[1]
    ])


def f_real(x, t):
    res = np.zeros_like(x)
    res[0] = 10 * (x[1] - x[0])
    res[1] = x[0] * (28 - x[2]) - x[1]
    res[2] = x[0] * x[1] - 8 / 3 * x[2]
    
    return res

t0 = 0.
tend = 2000 # 3000
dt = 0.02 # 0.01
x0 = np.array([1.5, -1.5, 20.])
t_span = np.arange(t0, tend, dt)

# Simulations
systems = {
    'lorenz_gt': f_real,
    'example1': f_example1,
    'example2': f_example2,
    'example3': f_example3,
    'example4': f_example4,
    'example5': f_example5,
    'example6': f_example6,
    'example7': f_example7,
    'example8': f_example8,
    'example9': f_example9,
}

solutions = {'t_span': t_span, 'x0': x0}

for name, func in systems.items():
    solutions[name] = odeint(func, x0, t_span)

# with open('./save_data/lorenz_solutions_additional.pkl', 'wb') as f:
#     pickle.dump(solutions, f)


import utils

kl_divergence_1 = utils.kl_divergence_attractors(solutions['lorenz_gt'][:, 0:3], solutions['example1'][:, 0:3], drop_zero=True)
kl_divergence_2 = utils.kl_divergence_attractors(solutions['lorenz_gt'][:, 0:3], solutions['example2'][:, 0:3], drop_zero=True)
kl_divergence_3 = utils.kl_divergence_attractors(solutions['lorenz_gt'][:, 0:3], solutions['example3'][:, 0:3], drop_zero=True)
kl_divergence_4 = utils.kl_divergence_attractors(solutions['lorenz_gt'][:, 0:3], solutions['example4'][:, 0:3], drop_zero=True)
kl_divergence_5 = utils.kl_divergence_attractors(solutions['lorenz_gt'][:, 0:3], solutions['example5'][:, 0:3], drop_zero=True)
kl_divergence_6 = utils.kl_divergence_attractors(solutions['lorenz_gt'][:, 0:3], solutions['example6'][:, 0:3], drop_zero=True)
kl_divergence_7 = utils.kl_divergence_attractors(solutions['lorenz_gt'][:, 0:3], solutions['example7'][:, 0:3], drop_zero=True)
kl_divergence_8 = utils.kl_divergence_attractors(solutions['lorenz_gt'][:, 0:3], solutions['example8'][:, 0:3], drop_zero=True)
kl_divergence_9 = utils.kl_divergence_attractors(solutions['lorenz_gt'][:, 0:3], solutions['example9'][:, 0:3], drop_zero=True)

print(f"kl_divergence_1: {kl_divergence_1}")
print(f"kl_divergence_2: {kl_divergence_2}")
print(f"kl_divergence_3: {kl_divergence_3}")
print(f"kl_divergence_4: {kl_divergence_4}")
print(f"kl_divergence_5: {kl_divergence_5}")
print(f"kl_divergence_6: {kl_divergence_6}")
print(f"kl_divergence_7: {kl_divergence_7}")
print(f"kl_divergence_8: {kl_divergence_8}")
print(f"kl_divergence_9: {kl_divergence_9}")
















































