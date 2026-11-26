"""
Passive torque at every sample: the target is measured torque minus this passive torque.

It uses the passive recordings (maps) of the trial's own session; never the torque of the trials it is evaluated
on.

legacy        old pipeline's map (20 Nm rules), looked up at each sample's angle (YES: worksheet values)
static        map from static holds only

The setting 'none' (total torque, nothing subtracted) is handled in prepare.prepare.
"""

import numpy as np

from ML.preprocessing.passive_map import lookup_passive_torque

KINDS = ('legacy', 'static')


def map_lookup(entries, pos):
    """Pipeline-identical map interpolation/extrapolation at each sample."""
    if not entries:
        return np.zeros(len(pos))
    return np.array([lookup_passive_torque(entries, float(p)) for p in pos])


def passive_torque(kind, bundle, rec):
    """Passive torque (Nm) at every sample of one trial, from the ``kind`` map of the trial's own session."""
    if kind not in KINDS:
        raise ValueError(f'passive kind must be one of {KINDS}, got {kind!r}')
    return map_lookup(bundle['maps'][kind][rec['session']], rec['pos'])
