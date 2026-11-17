"""
Shared configuration for the EMG-to-torque modelling pipeline.

Central source of truth for subject registry, hyperparameters,
and feature-layout constants used by train.py and cross_subject.py.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
ML_DIR    = os.path.dirname(__file__)
DATA_DIR  = os.path.join(ML_DIR, 'subjects')
MODEL_DIR = os.path.join(ML_DIR, 'checkpoints')
PLOT_DIR  = os.path.join(ML_DIR, 'plots')

# ── Subject registry ─────────────────────────────────────────────────────────
# Maps a short name to (flb_filename, subject_id_for_channel_ordering)
SUBJECTS = {
    'IES': ('IES_081024.flb', 'IES01'),   # S1 — non-standard channel order
    'YES': ('YES_281124.flb', 'default'),  # S3
    'EG':  ('EG_121224.flb',  'default'),  # S4
    'CH':  ('CH_151224.flb',  'default'),  # S5
    'JM':  ('JM_140325.flb',  'default'),  # S6
    'HM':  ('HM_110425.flb',  'IES01'),   # S7 — same non-standard channel order as IES01
}

# ── Per-subject trial index overrides ────────────────────────────────────────
# Used when auto-detection from comment strings is unreliable.
# Values are 1-indexed trial numbers (as passed to build_dataset).
# YES: passive/dynamic trials lack the "passive" keyword → must specify explicitly.
TRIAL_OVERRIDES = {
    'YES': {
        'test':   [11, 13, 15, 17, 19, 21, 23, 25],
        'retest': [27, 29, 31, 33, 35, 37, 39, 41],
    },
}

# ── Per-subject passive torque overrides ─────────────────────────────────────
# Hardcoded (position_rad, passive_torque_Nm) pairs read from the experimenter
# worksheet.  Used when auto-detection from the FLB passive trials is unreliable
# (e.g. most passive trials are movement recordings, not static holds).
#
# YES: testing order was random sequence [6,1,7,4,5,8,3,2].
# Worksheet rows P1–P8 list trials in that testing order, so the mapping is:
#   row P1→pos P6(−0.278), P2→P1(+0.245), P3→P7(−0.382), P4→P4(−0.069),
#   row P5→pos P5(−0.174), P6→P8(−0.487), P7→P3(+0.031), P8→P2(+0.137)
PASSIVE_TORQUE_OVERRIDES = {
    'YES': {
        'test': [
            (-0.487, +7.9),
            (-0.382, +4.3),
            (-0.278, +2.6),
            (-0.174, +1.8),
            (-0.069, +0.35),
            (+0.031, -2.6),
            (+0.137, -3.5),
            (+0.245, -10.1),
        ],
        'retest': [
            (-0.487, +10.6),
            (-0.382, +6.7),
            (-0.278, +3.5),
            (-0.174, +2.4),
            (-0.069, +0.73),
            (+0.031, -0.38),
            (+0.137, -2.5),
            (+0.245, -5.0),
        ],
    },
}

# ── Training hyperparameters ─────────────────────────────────────────────────
LEARNING_RATE  = 0.003
LR_FACTOR      = 0.7
LR_PATIENCE    = 10
LR_MIN         = 1e-6
EARLY_PATIENCE = 25
BATCH_SIZE     = 8
EPOCHS         = 200

# ── Model / feature layout ──────────────────────────────────────────────────
N_STEPS    = 20          # window length in timesteps at 100 Hz (200 ms)
N_FEATURES = 5           # [MG_env, LG_env, SOL_env, TA_env, position]

POSITION_FEATURE_INDEX = 4   # column index of position in feature array
TIME_STEP_S            = 0.01  # seconds per window step (stride=1 at 100 Hz)
