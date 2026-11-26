"""
Passive-torque models: the target is measured torque minus the model's passive torque.

A model uses the passive recordings (maps) of the trial's own session; never the torque of the trials it is
evaluated on.

legacy        old pipeline's map (20 Nm rules), looked up at each sample's angle (YES: worksheet values)
static        map from static holds only

The setting 'none' (total torque, nothing subtracted) is handled in run.prepare.
"""

import numpy as np

from ML.preprocessing.passive_map import lookup_passive_torque

KINDS = ('legacy', 'static')


def map_lookup(entries, pos):
    """Pipeline-identical map interpolation/extrapolation at each sample."""
    if not entries:
        return np.zeros(len(pos))
    return np.array([lookup_passive_torque(entries, float(p)) for p in pos])


class PassiveModel:
    """P(t) for any trial from its session's passive map (kind: see the module docstring)."""

    def __init__(self, kind, bundle):
        if kind not in KINDS:
            raise ValueError(f'passive kind must be one of {KINDS}, got {kind!r}')
        self.kind = kind
        self.bundle = bundle

    def predict(self, rec):
        """P(t) for one trial (any session)."""
        return map_lookup(self.bundle['maps'][self.kind][rec['session']], rec['pos'])

    def describe(self):
        return {'kind': self.kind}
