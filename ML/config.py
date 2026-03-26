"""
Shared configuration for the EMG-to-torque modelling pipeline.

Central source of truth for subject registry, hyperparameters,
and feature-layout constants used by train.py and cross_subject.py.
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
    'HM':  ('HM_110425.flb',  'IES01'),
    'EG':  ('EG_121224.flb',  'default'),
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
