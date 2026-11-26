"""Paths, signal constants and training protocols of the experiment pipeline.

The recordings folder and the sampling rates are in ML.config; RESEARCH_CACHE_DIR moves the processed-data cache.
"""

import os

CACHE_DIR = os.environ.get('RESEARCH_CACHE_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '_cache'))

WINDOW = 20              # steps (200 ms) per input window
TRIM_START = 1000        # raw samples dropped at the start of a trial (filter start-up)
TRAIN_FRAC = 0.8         # first 80 % of each training trial = train, rest = validation

# 'orig' = the original pipeline's settings (EXPERIMENTS/baseline/original_code/ML/config.py); 'fast' = batch 64 for screening; 'fast400' = up to 400 epochs.
PROTOCOLS = {
    'orig': dict(batch=8, lr=0.003, lr_factor=0.7, lr_patience=10, early=25, epochs=200),
    'fast': dict(batch=64, lr=0.003, lr_factor=0.5, lr_patience=6, early=15, epochs=150),
    'fast400': dict(batch=64, lr=0.003, lr_factor=0.5, lr_patience=6, early=15, epochs=400),
    # study v2 (every round): early stopping under a 400-epoch cap; batch and lr come from each config
    'study400': dict(batch=64, lr=0.003, lr_factor=0.5, lr_patience=6, early=15, epochs=400),
}
