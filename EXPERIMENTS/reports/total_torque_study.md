# Total-torque study (TT01–TT04)

Status: **design, checks and benchmark done; study not yet run.** Results sections are filled in only from
completed, matched runs. Nothing here claims that total torque or any normalisation method is better.

## Question

The active-torque experiments (X00f … D01, D02) train on `measured total torque − estimated passive torque`.
Passive-table errors and session-dependent passive levels may limit them. This study trains on
**measured total torque** instead, so no passive table enters the target, and compares four EMG amplitude
normalisations under that target. It is **exploratory**: the same two sessions of the same six people
already shaped earlier design choices, so this is not a fresh, independent confirmation.

## Fixed design (identical in all four experiments)

| item | value | where in code |
|---|---|---|
| target | processed measured torque (1 kHz → 1 s start-up trim → zero-phase FIR decimation ×10 to 100 Hz), **no passive subtraction** | `ML/research/data.py::_process_active`, `ML/research/run.py::prepare` (`passive='none'`) |
| inputs | 4 EMG envelopes (GM, GL, SOL, TA; band-pass 30–300 Hz, rectify, 2 Hz low-pass, then decimated) + ankle angle | `data.py`, `splits.windows` |
| window | 200 ms = 20 samples at 100 Hz, stride 1; target = torque at the window's last sample | `splits.py` |
| split | source session: first 80 % of the windows of every trial = train, rest = validation (last validation window per trial dropped, as in the original split); target session: every window of every trial = test | `splits.build`, `run_experiment.drop_last_val_window` |
| balanced positions | D01's `balance='scale'`: positions = training trial angles merged within 0.02 rad; train on `target / SD_k`, `SD_k` = SD of the **training-part** targets of position k, floored at **0.1 Nm** (near-zero spread); predictions × `SD_k` of the nearest training position → Nm before any scoring | `ML/research/train.py::fit_predict` |
| model | LSTM 8 → LSTM 8 (tanh), dropout 0.3 after each, Dense 1 | `run_experiment.make_builder` |
| training | Nadam 0.003, batch 8, MSE, ReduceLROnPlateau (×0.7, patience 10), EarlyStopping (patience 25, restore best) on **source-session validation loss**, ≤ 200 epochs | `ML/research/config.py` `PROTOCOLS['orig']` |
| auto-gain | **off** (no moving or adaptive amplitude normalisation) | configs: `emg` ≠ `agc*` |
| seeds | matched: seed k is the same for every method, subject and direction | `run_queue.py` |

The target is never selected, tuned or stopped on target-session torque. Every training choice above
is D01's.

## Normalisation variants

| id | `emg` | scale per muscle | data used for the scale |
|---|---|---|---|
| TT01 | `raw` | none: envelopes in original units (mV) | — |
| TT02 | `trainmax_tr` | maximum of the 100 Hz envelope, positions pooled | training part of the source session only; applied unchanged to validation and to the other session |
| TT03 | `trainp99` | 99th percentile, positions pooled | as TT02 |
| TT04 | `session_p99` | 99th percentile per session, positions pooled | source session: its training part; target session: **all of its active-trial EMG, no torque** |

**TT04 is offline, unlabelled target-session adaptation.** It needs the whole target recording before
predicting and assumes that effort ranges are reasonably comparable between sessions. TT02 and TT03 are
fixed: the scale is set once from training data and never changes.

Not used, on purpose: per-position normalisation, per-trial mean removal, and clipping above 1. MVC
normalisation was not included: the recordings contain MVC trials, but it is not established that MVC was
repeated in both sessions.

Zero denominators: a fixed scale that is not finite or ≤ 1e-9 is replaced by 1.0 and listed under
`guarded_channels` in each run's JSON (`emg_norm._safe_scale`); `session_p99` keeps its existing 1e-9 floor.
All scales are saved in every run's JSON (`details.emg`).

## Checks run before the study (`EXPERIMENTS/code/verify_total_torque.py`)

On YES, JM and EG, both directions, all four configurations (`EXPERIMENTS/results/total_torque_checks.json`):

1. every train / validation / test target equals the processed measured torque at the window's last sample,
   and differs from the D01 active target;
2. the last input step of every window is the target sample;
3. TT02/TT03 scales equal a direct recomputation from the training part and do not change when the
   validation part and the whole other session are multiplied by 7; TT04's source scale ignores them and its
   target scale follows target-session EMG only;
4. every feature is envelope / one constant per channel (no auto-gain);
5. every training position has training windows and a finite SD; a network that always outputs 1.0 returns
   exactly `SD_k` of each test window's position (predictions are restored to Nm).

`run_experiment.py` also refuses to save a TT run whose target differs from the measured torque read
independently from the recording.

The research framework imports a pinned copy of the preprocessing it was developed with
(`ML/research/d01_preprocessing/`, see its `__init__.py`). `EXPERIMENTS/code/check_d01_pin.py` showed that
this branch builds D01's windowed arrays bit-identically to `experimentation-dorsiflexion` (YES, JM, IES,
both directions, 54 arrays each).

## Benchmark of this computer (measured 1 Nov 2026, `EXPERIMENTS/results/total_torque_benchmark.json`)

Hardware: Intel i7-14700F (20 cores = 8 P + 12 E, 28 threads), 31.8 GB RAM, NVIDIA RTX 5070 12 GB.
TensorFlow 2.21.0 on native Windows is a CPU-only build (`is_built_with_cuda() == False`, no visible GPU):
**the GPU is not used.** (GPU training would need WSL2; not pursued, the model is tiny and batch 8.)

Representative run: YES, S1 → S2, TT02 (total torque, 200 ms, balanced positions), 56 896 train /
14 224 validation / 71 128 test windows (identical counts for every subject and direction).

| measured | value |
|---|---|
| read + filter one .flb into the cache (cold) | 2.8 s |
| build windows from the cache (every job) | 0.25 s |
| 1 job, 1 / 2 / 4 TF threads | 19.9 / 17.2 / 17.9 s per epoch |
| model build + prediction overhead | ≈ 1–2.5 s |
| 8 / 16 / 24 concurrent 1-thread jobs | 29.5 / 43.6 / 56.4 s per epoch each → ≈ 980 / 1 320 / 1 530 epochs per hour in total |
| peak memory per job | 584 MB (24 jobs ≈ 14 GB) |

Projection (not measured): epochs per run are not known until the runs stop early. D01 + C01 (same
training rules, active target, 78 runs) needed mean 61.9, median 58.5, 90th percentile 93, max 132 epochs;
the total target may stop differently. With that mean, 24 parallel jobs:

| plan | runs | epochs (projected) | time at 24 jobs |
|---|---|---|---|
| pilot: YES, EG, JM × 4 methods × 2 directions × seed 0 | 24 | ≈ 1 500 | ≈ 1–2 h (one wave; longest run sets the end) |
| full study, 5 seeds | 240 | ≈ 14 900 | ≈ 10 h (+ ≈ 1 h straggler tail) |
| full study, 10 seeds | 480 | ≈ 29 700 | ≈ 20 h (+ ≈ 1 h) |
| optional: D01 re-run with saved predictions, 5 seeds × 2 directions | 60 | ≈ 3 700 | ≈ 2.5 h |

At 16 jobs (computer stays usable) the full 5-seed study is ≈ 11–12 h, 10 seeds ≈ 23 h.
