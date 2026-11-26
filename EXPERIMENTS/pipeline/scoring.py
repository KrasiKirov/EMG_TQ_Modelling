"""Normalising scale for scores computed from saved predictions (used by analysis/confirm.py and tuning/select.py)."""
import numpy as np


def person_scale(y_measured):
    """Normalising scale of one person and target: 99th percentile of |measured target| (never a prediction)."""
    return float(np.percentile(np.abs(np.asarray(y_measured, float)), 99))
