"""
Frozen copy of the preprocessing used by every ML.research / EXPERIMENTS run
(X00f ... D01, D02 and the total-torque study).

These files are byte-identical to ML/preprocessing/{flb_reader, emg_envelope,
dataset_builder, active_torque, emg_agc}.py, ML/models/lstm.py and
ML/config.py on branch ``experimentation-dorsiflexion`` (commit c86c857).
They were pinned here when that branch was merged with ``experimentation``,
whose ML/ package changed the same modules (for example the passive-torque
bound in dataset_builder: 20 Nm here vs 10 Nm there, and extract_envelope's
return value).  Keeping this copy means the research framework builds exactly
the data its saved results were trained on, while ML/ keeps the audited
pipeline of the experimentation branch.  Do not edit; equivalence with the
original branch is checked by EXPERIMENTS/code/check_d01_pin.py.
"""
