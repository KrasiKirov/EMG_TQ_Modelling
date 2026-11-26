"""
Passive torque at every sample: the target is measured torque minus this passive torque.

It is looked up at each sample's angle in the static-hold map of the trial's own session (ML.research.data), never
taken from the torque of the trials it is evaluated on. The setting 'none' (total torque, nothing subtracted) is
handled in prepare.prepare.
"""

import numpy as np

from ML.preprocessing.passive_map import lookup_passive_torque

KINDS = ('static',)


def map_lookup(entries, pos):
    """Map interpolation / extrapolation at each sample (0 everywhere for an empty map)."""
    if not entries:
        return np.zeros(len(pos))
    return np.array([lookup_passive_torque(entries, float(p)) for p in pos])


def passive_torque(bundle, rec):
    """Passive torque (Nm) at every sample of one trial, from the static map of the trial's own session."""
    return map_lookup(bundle['maps']['static'][rec['session']], rec['pos'])
