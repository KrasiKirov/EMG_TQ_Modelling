# ML/research: cross-session study of the plantarflexed-angle problem

This package is **added next to** the existing pipeline. It imports
`ML.preprocessing`, `ML.models` and `ML.evaluation` and does not change them.
Delete the folder and the rest of the repo behaves exactly as before.

## Question

Can we predict **voluntary (active) ankle torque** from 4 EMG channels at every
ankle angle? In particular, can we do it at the most plantarflexed angles,
where the original pipeline had negative R², **when the model is tested on a
session it never saw**? Subjects: YES, JM (worst before), and EG (a third
subject not used to design the earlier fixes).

## Modules

| file | what it does |
|---|---|
| `config.py` | paths, subjects, signal constants, training protocols |
| `data.py` | loads a subject into per-trial records (EMG envelopes, angle, torque, the passive hold before each trial, MVC trials, passive maps), cached |
| `passive.py` | passive-torque models (old map, static map, gain-corrected, hold-based, relaxation/drift) |
| `truth.py` | the **common reference truth**: measured torque minus a passive baseline drawn through the EMG-detected rest periods of the same trial; rest/active metrics |
| `emg_norm.py` | EMG normalisation (training max, causal auto-gain, MVC, per-session) |
| `splits.py` | protocols and sliding windows |
| `models.py` | LSTM, mixture-of-experts LSTM, optional EMG gain augmentation |
| `train.py` | training with per-angle loss balancing, relaxed-window weighting, ridge baseline |
| `transfer.py` | cross-subject pre-training (leave-one-subject-out) + fine-tuning |
| `experiments.py` | the named configurations E01…E32 and `FINAL` (each with the question it answers) |
| `run.py` / `queue.py` | run one job / many jobs in parallel |
| `analyze.py` | tables, cross-fitted selection, statistics, ensembles, calibration |
| `figures.py` | report figures |

## The recommended pipeline: `FINAL`

`FINAL` (report 2, section 10) combines:
- rest-anchored labels;
- causal auto-gain with report 1's "auto" rule (auto-gain off for subjects who push in both directions);
- the mixture-of-experts LSTM;
- the per-angle scaled loss.

It was run on all 6 subjects × 2 directions (seed 0, `results/final_check/`). In all 12 runs it gives exactly the same predictions as `E22`, or as `E32` for CH, the only subject who pushes both ways. A step-by-step explanation is in `notes/final_pipeline/final_code_pipeline.md`.

## Protocols (`splits.py`)

| name | train / validation | test |
|---|---|---|
| `S1toS2` | first 80 % / last 20 % of every session-1 trial | every session-2 trial (whole trial) |
| `S2toS1` | same with session 2 | every session-1 trial |
| `pooled` | first 60 % / 60–80 % of every trial of **both** sessions | last 20 % of every trial |

In `pooled`, a 20-sample (one-window) gap at each cut keeps any window from
spanning two parts. In the cross-session protocols the test session is never
seen in training, so no gap is needed; the train/validation cut has no gap,
the same as the original pipeline. `S1toS2` and `S2toS1` form a 2-fold
cross-session validation.

## Scoring

Each configuration has its own target (measured torque minus its own passive
model). Scoring each configuration against its own target would mean scoring
them against different truths. Every prediction is therefore also scored
against **one common reference truth** (`truth.py`), and this is the primary
result. The model never sees the reference truth as an input.

## Rules that prevent leakage

* Torque used to build training labels comes only from the training
  session (pooled: the train + validation part of each trial).
* EMG statistics come from the training session. The per-channel maximum
  uses the train + validation part, the same convention as the original
  pipeline (pooled: the first 80 % of each trial). The auto-gain priors use
  the training part only.
* `session_p99` (E26) uses unlabelled test-session EMG. It is
  **transductive**, flagged as such, and excluded from the causal selection.
* The model is chosen on validation loss (early stopping). The test set is
  never used for any choice inside a run.
* Configurations are compared by **cross-fitting** (`analyze.py`): the
  configuration reported on fold A is the one that was best on fold B. A
  leave-one-subject-out selection is also reported. Both still share the
  same 3 subjects, so they are not fully independent of the selection.
* Every configuration runs with seeds 0, 1 and 2, and results are the
  **mean over seeds** (or a seed ensemble, i.e. the average of the 3
  networks). No seed is ever picked.

## How to run

```bash
source ML/experiments/env.sh          # sets PYTHONPATH, EMG_DATA_DIR, $PY
# one job
$PY -m ML.research.run --study demo --subject YES --protocol S1toS2 --config FINAL --seed 0 --plot
# many jobs in parallel (skips finished or running ones, safe to relaunch)
$PY -m ML.research.queue --study screen --subjects YES JM EG --protocols S1toS2 S2toS1 \
    --configs E01_old E02_minimal E03_final --seeds 0 1 2 --workers 12 --threads 1
# tables, cross-fitted selection, statistics
$PY -m ML.research.analyze --study screen
# figures
$PY -m ML.research.figures --study screen --out notes/report2/figures
```

## Outputs

`results/<study>/<subject>/<protocol>/<config>/` contains:

* `fast_seed<k>.json`: config, passive-model coefficients, training info, and
  metrics per angle and overall against the configuration's own target
* `fast_seed<k>_truth.json`: the same metrics against the reference truth,
  plus calibration curves (cache written by `analyze.py`)
* `fast_seed<k>.npz`: test truth, predictions and window metadata
* `full_trial_inspection_<S1|S2>_fast_seed0.png`: every test trial, target
  vs prediction

`results/<study>/_summary/` holds `summary.json` and `tables.md`.
`results/_logs/` holds one console log per job (git-ignored).
`results/superseded_val_weights/` holds runs trained before a bug fix, kept
only for traceability.
