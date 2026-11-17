"""Scoring metrics (used by the presentation figures)."""

import numpy as np


def metrics(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    e = p - y
    ss_tot = np.sum((y - y.mean()) ** 2)
    rng = np.percentile(y, 98) - np.percentile(y, 2)
    return {
        'r2': float(1 - np.sum(e ** 2) / ss_tot) if ss_tot > 0 else float('nan'),
        'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))),
        'bias': float(np.mean(e)),
        'vaf': float((1 - np.var(e) / np.var(y)) * 100) if np.var(y) > 0 else float('nan'),
        'nrmse': float(np.sqrt(np.mean(e ** 2)) / rng) if rng > 0 else float('nan'),
        'std_true': float(np.std(y)), 'n': int(len(y)),
    }
