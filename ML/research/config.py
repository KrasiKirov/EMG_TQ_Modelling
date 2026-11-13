"""
Configuration for the ML.research study.

Paths can be overridden with environment variables so the package can run from
a git worktree while the (git-ignored) recordings stay in the main checkout:

    EMG_DATA_DIR         folder with the .flb recordings (read by ML.config)
    RESEARCH_CACHE_DIR   processed-data cache   (default ML/research/_cache)
    RESEARCH_RESULTS_DIR results root           (default ML/research/results)
"""

import os

RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get('RESEARCH_CACHE_DIR', os.path.join(RESEARCH_DIR, '_cache'))
RESULTS_DIR = os.environ.get('RESEARCH_RESULTS_DIR', os.path.join(RESEARCH_DIR, 'results'))
LOG_DIR = os.path.join(RESULTS_DIR, '_logs')

# Subjects of this study: the two subjects with the worst plantarflexed-angle
# R² (YES, JM) plus EG, a subject not used to design the earlier fixes, whose
# most plantarflexed angle was also weak in the original pipeline.
STUDY_SUBJECTS = ['YES', 'JM', 'EG']

# Every subject with recordings (used as SOURCE data for cross-subject
# pre-training in transfer.py; the target subject is always left out).
ALL_SUBJECTS = ['IES', 'YES', 'JM', 'EG', 'CH', 'HM']

# Signal layout (same as the main pipeline)
FS_RAW = 1000.0          # Hz, recordings
DOWNSAMPLE = 10          # 1000 Hz -> 100 Hz
FS = FS_RAW / DOWNSAMPLE
WINDOW = 20              # steps (200 ms) per input window
TRIM_START = 1000        # raw samples dropped at the start of a trial (filter start-up)
TRAIN_FRAC = 0.8         # first 80 % of each training trial = train, rest = validation

# Training protocols.  'orig' = the pipeline defaults (ML/config.py); 'fast'
# = larger batch for screening many configurations (same optimiser and loss).
PROTOCOLS = {
    'orig': dict(batch=8, lr=0.003, lr_factor=0.7, lr_patience=10, early=25, epochs=200),
    'fast': dict(batch=64, lr=0.003, lr_factor=0.5, lr_patience=6, early=15, epochs=150),
}
