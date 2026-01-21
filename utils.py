# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:37:43 2023

@author: zmzhai
"""

import os
import copy
import torch
import pickle
import random
import numpy as np
import pandas as pd

def kl_divergence_attractors(gt, pred, bins=100, direction='gt||pred', drop_zero=False):
    """
    KL between two 3D attractors via 3D histogram.
    - Each dataset is normalized independently to [0,1] per dimension.
    - 'one node exists in every box' => add +1 to each bin, then normalize.
    - direction: 'gt||pred' (default) or 'pred||gt'
    """
    gt = np.asarray(gt, float); pred = np.asarray(pred, float)
    assert gt.ndim == 2 and gt.shape[1] == 3
    assert pred.ndim == 2 and pred.shape[1] == 3

    def norm01(X):
        lo = X.min(axis=0); span = np.maximum(X.max(axis=0) - lo, 1e-12)
        return (X - lo) / span

    gt_n, pred_n = norm01(gt), norm01(pred)

    if isinstance(bins, int):
        bins = (bins, bins, bins)
    edges = [np.linspace(0.0, 1.0, b + 1) for b in bins]

    H_gt, _   = np.histogramdd(gt_n,   bins=edges)
    H_pred, _ = np.histogramdd(pred_n, bins=edges)

    p = H_gt.ravel().astype(float)
    q = H_pred.ravel().astype(float)
    K = p.size
    Np, Nq = p.sum(), q.sum()

    if drop_zero:
        mask = (p > 0) & (q > 0)
        if not np.any(mask): return np.inf
        p = p[mask]; q = q[mask]
        p /= p.sum(); q /= q.sum()
        
        aaa = 1
    else:
        p = (H_gt.ravel().astype(float) + 1.0)
        q = (H_pred.ravel().astype(float) + 1.0)
        p /= p.sum(); q /= q.sum()

    if direction == 'gt||pred':
        return float(np.sum(p * (np.log(p) - np.log(q))))
    elif direction == 'pred||gt':
        return float(np.sum(q * (np.log(q) - np.log(p))))
    else:
        raise ValueError("direction must be 'gt||pred' or 'pred||gt'")









