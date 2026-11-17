# ML/research: data and training for the experiment pipeline

`EXPERIMENTS/pipeline/run.py` builds every experiment's data and model from this package. It imports
`ML.preprocessing`, `ML.models` and `ML.evaluation` and does not change them.

| file | what it does |
|---|---|
| `config.py` | cache path, signal constants, training protocols |
| `data.py` | loads a subject into per-trial records (EMG envelopes, angle, torque, the passive hold before each trial, MVC trials, passive maps), cached |
| `passive.py` | passive-torque models (old map, static map, gain-corrected, hold-based, relaxation/drift) and EMG rest detection |
| `truth.py` | the in-trial passive baseline through the EMG-detected rest periods (`restanchor` target, rest flags) |
| `emg_norm.py` | EMG normalisation (training max, causal auto-gain, MVC, per-session) |
| `splits.py` | protocols (`S1toS2`, `S2toS1`, `pooled`), sliding windows and whole-trial streams |
| `models.py` | LSTM and mixture-of-experts LSTM builders, optional EMG gain augmentation |
| `train.py` | Keras training with per-angle loss balancing and relaxed-window weighting; ridge baseline |
| `stateful.py` | LSTM trained on whole trials with the state carried between chunks |
| `run.py` | `prepare()`: everything up to the windowed arrays |

## Protocols (`splits.py`)

| name | train / validation | test |
|---|---|---|
| `S1toS2` | first 80 % / last 20 % of every session-1 trial | every session-2 trial (whole trial) |
| `S2toS1` | same with session 2 | every session-1 trial |
| `pooled` | first 60 % / 60–80 % of every trial of **both** sessions | last 20 % of every trial |

## Rules that prevent leakage

* Torque used to build training labels comes only from the training session (pooled: the train + validation
  part of each trial).
* EMG statistics come from the training session; the per-channel maximum uses the train + validation part, as
  in the original pipeline.
* `session_p99` uses unlabelled test-session EMG; it is transductive and flagged as such.
* The model is chosen on validation loss (early stopping); the test set is never used for any choice in a run.

`results/` holds the round-2 study's saved runs (October 2026), kept as records; the scripts that produced and
summarised them were removed and are in the git history.
