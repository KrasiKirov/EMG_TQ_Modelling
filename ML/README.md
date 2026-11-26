# ML: from REKLAB recordings to ankle-torque predictions

This package turns one participant's `.flb` recording into model inputs, and trains the models that predict ankle
torque from EMG. It never reads experiment specs or writes results: `EXPERIMENTS/pipeline` does that and calls into
this package.

## Layout

| module | what it does |
|---|---|
| `config.py` | participants and their `.flb` files, folders, sampling rates, trims, split fraction, the training schedule (`TRAINING`), per-participant trial and passive-torque overrides |
| **`preprocessing/`** | **recording → signals (1 kHz)** |
| `preprocessing/flb_reader.py` | reads an `.flb` file into one DataFrame per trial: `time`, `position` (rad), `torque` (Nm), `gm`, `gl`, `sol`, `ta` (EMG, mV) |
| `preprocessing/trials.py` | sorts trials into MVC / passive / active by their comment; splits a passive trial at device steps |
| `preprocessing/emg_envelope.py` | EMG envelope of one channel: band-pass 30–300 Hz, rectify, 2 Hz low-pass, forward-only (only past EMG, as in a real-time device) |
| `preprocessing/passive_map.py` | passive-torque map (angle → torque) from the static passive holds, and the lookup between its angles |
| `preprocessing/rest.py` | finds rest periods in a trial from the EMG alone |
| **`research/`** | **signals → windowed arrays → trained model** |
| `research/data.py` | `load_subject`: one participant as a cached "bundle" of 100 Hz trial records and passive maps |
| `research/emg_norm.py` | EMG scaling: each channel ÷ its maximum over the training part of session 1 |
| `research/passive.py` | passive torque at every sample, from the trial's own session map |
| `research/rest_flags.py` | rest flag of every sample of every active trial (stored with each window) |
| `research/splits.py` | session split (`S1toS2`), sliding windows, whole-trial streams, the targets shared by every window length |
| `research/prepare.py` | `prepare`: runs everything above and returns the train / validation / test arrays |
| `research/train.py` | `fit_predict`: trains the windowed LSTM and predicts session 2 (setting `state: reset`) |
| `research/stateful.py` | the LSTM trained on whole trials with its state carried between chunks (setting `state: carry`) |
| **`models/`** | **network and baseline definitions** |
| `models/lstm.py` | `build_lstm`: stacked LSTM layers with dropout, then one linear output (torque, Nm) |
| `models/baselines.py` | `LaggedRidge`: study v2 baseline A1, ridge on lagged envelopes |

`research/stateful.py` builds its own network (`build_seq`) because the carried state needs the states as inputs
and outputs; it is kept apart from `models/lstm.py` on purpose.

## Data flow

One participant, settings of the study v2 model A0 (spec `EXPERIMENTS/configs/FE_A0.json`):

1. **Read** the `.flb` file (`read_flb`): 1 kHz, one DataFrame per trial.
2. **Sort the trials** (`classify_trials`, `ML.config.TRIAL_OVERRIDES`): active trials of session 1 (lab "test")
   and session 2 ("retest"), and passive trials.
3. **Envelopes** of the 4 EMG channels: band-pass 30–300 Hz, rectify, 2 Hz low-pass (2nd order, forward-only).
4. **100 Hz:** inputs are every 10th sample; the torque is down-sampled with a zero-phase FIR filter. The first 1 s
   and last 1 s of each trial are dropped. Each active trial becomes a record with `env_raw` (T, 4), `pos` (T,) rad,
   `tq` (T,) Nm, `t` (T,) s, its session and mean angle.
5. **Target** = measured torque − passive torque (setting `passive: static`). The passive torque is looked up at each
   sample's angle in the map of that trial's own session, built from its static relaxed holds. IES has no usable
   holds: its map comes from `ML.config.METADATA_PASSIVE`. `passive: none` keeps the total torque.
6. **EMG scaling:** each channel ÷ its largest value in the training part of the session-1 trials.
7. **Split** (`S1toS2`): in every session-1 trial the first 80 % of windows train; validation starts one window
   length later, so no validation window reuses a training sample; the pipeline drops the last validation window.
   Every session-2 trial is test data.
8. **Windows:** 20 consecutive samples (0.2 s) × [GM, GL, SOL, TA, angle] = (20, 5); the target is the torque at
   the last sample. A new window starts at every sample.
9. **Model:** A0 is a 64-unit, 1-layer LSTM trained on whole trials in chunks of 200 samples with its state
   carried between chunks (`stateful.py`). Nadam with gradient clipping (norm 1), MSE loss, learning rate × 0.5
   after 6 flat epochs, early stopping after 15, at most 400 epochs (`ML.config.TRAINING`), samples in time order
   (no shuffling).
10. **Output:** torque predictions (Nm) for every sample of session 2 that a window ends on, plus the validation
    predictions.

A spec can change only the passive target, window length, state, chunk, network size, dropout, learning
rate, batch and model (LSTM or lagged ridge); `EXPERIMENTS/pipeline/settings.py` lists the defaults and allowed
values, and `EXPERIMENTS/configs/TEMPLATE_reference.json` explains each one. Everything else above is fixed.

## Running

From the repository root, with the environment in `requirements-lock.txt`:

```sh
python -m EXPERIMENTS.pipeline.run --exp FE_A0 --subject JM --seed 10
```

- **Recordings:** `ML/subjects/` (not in git), or the folder named by `EMG_DATA_DIR`.
- **Cache:** processed bundles are cached in `ML/research/_cache/`, or in `RESEARCH_CACHE_DIR`. Delete the cache
  after changing anything in `research/data.py` or `preprocessing/` that changes a bundle, or raise
  `CACHE_VERSION` in `data.py`.
- **Tests:** `python -m pytest`; they use a synthetic participant and need no recordings.

## History

Options and models no longer used were removed in November 2026. Tag `pre-cleanup-2026-11` keeps the code before the
reorganization, tag `study-v2-final` keeps the settings retired first (overlap, per-angle loss scaling, input step),
and tag `older-options-2026-11-26` keeps the settings and the 84 specs that only the experiments before study v2
used, as well as the zero-phase envelope and its descriptive study v2 arm FE_A0_zerophase (whose saved runs stay).
`EXPERIMENTS/pipeline/settings.py` lists what was retired and when.
