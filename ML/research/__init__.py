"""
ML.research — experiment package for the plantarflexed-angle study (v2).

This package is APPENDED to the project: it imports the existing pipeline
(``ML.preprocessing``, ``ML.models``, ``ML.evaluation``) and never modifies
it.  Everything needed to reproduce the study lives here:

    config.py       paths, study subjects, training protocols
    data.py         load a subject once: active trials (both sessions), the
                    passive "before-holds", MVC trials -> 100 Hz records
    passive.py      passive-torque models (old map, static holds, gain
                    correction, time-varying drift model, hold-based levels)
    truth.py        the common reference truth used to score every
                    configuration (in-trial passive baseline through the
                    EMG-detected rest periods) + rest/active metrics
    emg_norm.py     EMG normalisations (training max, MVC, causal auto-gain,
                    per-session)
    splits.py       evaluation protocols (cross-session folds, pooled
                    sessions) and sliding windows
    models.py       model builders (wrappers around ML.models.lstm) and the
                    EMG gain-augmentation layer
    train.py        Keras training with per-angle loss balancing, relaxed-
                    window weighting, ridge-regression baseline
    transfer.py     cross-subject pre-training (leave-one-subject-out)
    evaluate.py     metrics per angle, recalibration, trial-inspection plots
    experiments.py  the registry of named configurations (what each tests)
    run.py          CLI: one (subject, protocol fold, config, seed) job
    queue.py        run many jobs in parallel on the CPU
    analyze.py      aggregate results -> tables, cross-fitted selection,
                    statistics, ensembles, calibration curves
    tables.py       markdown tables for the report (from summary.json)
    figures.py      report figures

See README.md for the protocol and how to run it.
"""
