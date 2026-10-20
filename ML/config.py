"""
Shared configuration for the EMG-to-torque modelling pipeline.

Central source of truth for subject registry, hyperparameters,
and feature-layout constants used across the ML pipeline.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
ML_DIR    = os.path.dirname(__file__)
DATA_DIR  = os.path.join(ML_DIR, 'data')
MODEL_DIR = os.path.join(ML_DIR, 'checkpoints')
PLOT_DIR  = os.path.join(ML_DIR, 'plots')

# ── Subject registry ─────────────────────────────────────────────────────────
# Maps a short name to (flb_filename, subject_id_for_channel_ordering)
SUBJECTS = {
    'HM':  ('HM_110425.flb',  'default'),
    'EG':  ('EG_121224.flb',  'default'),
    'IES': ('IES_081024.flb', 'default'),
    'JM':  ('JM_140325.flb',  'default'),
    'YES': ('YES_281124.flb', 'default'),
    # CH is stored with the source isometric recordings outside this repo.
    'CH':  ('/Users/krasi/Desktop/McGill/Capstone/isometric/CH01/CH_151224.flb',
            'default'),
}

# ── Training hyperparameters ─────────────────────────────────────────────────
LEARNING_RATE  = 0.003
LR_FACTOR      = 0.7
LR_PATIENCE    = 10
LR_MIN         = 1e-6
EARLY_PATIENCE = 25
BATCH_SIZE     = 8
EPOCHS         = 200
# Use the calibration map from the same acquisition session as each target:
# session 1 (``test``) for training/validation and session 2 (``retest``) for
# the held-out retest.  Pass ``'retest'`` explicitly to reproduce the
# historical shared-map pipeline.
PASSIVE_MAP_SESSION = 'session_specific'

# IES has no separately labelled passive trials in its FLB file.  These values
# are transcribed from data/experiment_metadata_IES.docx, which documents the
# passive torque measured during the MVC session. They are used only as an
# explicit fallback when a session-specific passive map cannot be built.
PASSIVE_MAP_FALLBACKS = {
    'IES': [
        (-0.6000, +1.60),
        (-0.4929, +1.36),
        (-0.3857, +0.60),
        (-0.2786, +0.40),
        (-0.1714, -1.00),
        (-0.0643, -2.40),
        (+0.0429, -5.60),
        (+0.1500, -5.80),
    ],
}
PASSIVE_MAP_FALLBACK_SOURCES = {
    'IES': 'documented MVC-session passive torque table',
}

# ── Model / feature layout ───────────────────────────────────────────────────
N_STEPS    = 50          # window length in timesteps at 100 Hz (500 ms)
N_FEATURES = 5           # [MG_env, LG_env, SOL_env, TA_env, position]
LSTM_UNITS = 8           # hidden units per LSTM layer (default)

POSITION_FEATURE_INDEX = 4   # column index of position in feature array
TIME_STEP_S            = 0.01  # seconds per window step (stride=1 at 100 Hz)

# ── Capacity sweep ────────────────────────────────────────────────────────────
# Unit counts compared in training/capacity_sweep.py
CAPACITY_SWEEP_UNITS = [8, 16, 32]
