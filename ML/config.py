"""Participants, folders, sampling rates, data-processing constants, the training schedule and per-participant
data overrides."""

import os

# Paths
ML_DIR    = os.path.dirname(__file__)
DATA_DIR  = os.environ.get('EMG_DATA_DIR', os.path.join(ML_DIR, 'subjects'))

# Sampling rates
FS_RAW = 1000.0            # Hz, recordings
DOWNSAMPLE = 10            # 1000 Hz -> 100 Hz
FS = FS_RAW / DOWNSAMPLE   # Hz, after downsampling

# Processed-data cache (one bundle per participant and envelope setting); RESEARCH_CACHE_DIR moves it
CACHE_DIR = os.environ.get('RESEARCH_CACHE_DIR',
                           os.path.join(os.path.dirname(os.path.abspath(__file__)), 'research', '_cache'))

# Windows and the session-1 split
WINDOW = 20              # steps (200 ms) per input window
TRIM_START = 1000        # raw samples dropped at the start of a trial (filter start-up)
TRIM_END_S = 1.0         # seconds dropped at the end of a trial
TRAIN_FRAC = 0.8         # first 80 % of each training trial = train, rest = validation

# Training schedule (study v2): Nadam, learning rate x lr_factor after lr_patience flat epochs, early stopping
# after `early` flat epochs under an `epochs` cap. A spec's lr and batch replace the defaults here.
TRAINING = dict(batch=64, lr=0.003, lr_factor=0.5, lr_patience=6, early=15, epochs=400)
MIN_LR = 1e-6            # the learning rate is never reduced below this
CLIPNORM = 1.0           # gradient-norm clipping of Nadam

# Subject registry
# Maps a participant's initials to (flb_filename, channel order for read_flb: 'IES01' or 'default')
SUBJECTS = {
    'IES': ('IES_081024.flb', 'IES01'),    # non-standard channel order
    'YES': ('YES_281124.flb', 'default'),
    'EG':  ('EG_121224.flb',  'default'),
    'CH':  ('CH_151224.flb',  'default'),
    'JM':  ('JM_140325.flb',  'default'),
    'HM':  ('HM_110425.flb',  'IES01'),    # same non-standard channel order as IES
}

# Per-subject trial index overrides
# Used when auto-detection from comment strings is unreliable.
# Values are 1-indexed trial numbers (trial_index + 1).
# YES: passive/dynamic trials lack the "passive" keyword → must specify explicitly.
TRIAL_OVERRIDES = {
    'YES': {
        'test':   [11, 13, 15, 17, 19, 21, 23, 25],
        'retest': [27, 29, 31, 33, 35, 37, 39, 41],
    },
}

# Per-subject passive torque overrides
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

# Passive torque from experiment metadata
# Used only when the subject has no relaxed-hold passive map.
# IES: no relaxed holds or usable passive trials were recorded; these are the passive torques noted at each
# position during the MVC trials (ML/subjects/experiment_metadata_IES.docx, recorded once for both sessions).
# (position_rad, passive_torque_Nm), same sign convention as the recordings (checked against IES's rest torque).
METADATA_PASSIVE = {
    'IES': [(-0.6, 1.6), (-0.4929, 1.36), (-0.3857, 0.6), (-0.2786, 0.4),
            (-0.1714, -1.0), (-0.0643, -2.4), (0.0429, -5.6), (0.15, -5.8)],
}
