# The "dorsiflexion" R² drop — investigation & fixes (2026-10-04)

Worktree branch `claude/dorsiflexion-r2-drop-341137` (nothing committed).
Starting point = the uncommitted state of the main checkout (6 subjects,
YES worksheet overrides, session-specific passive maps).

---

## TL;DR

* The failing angles are the most **plantarflexed** ones, not dorsiflexed.
  REKLAB convention (lab code: `maxDF = +0.25`, `maxPF = −0.50` rad):
  positive = dorsiflexion.  Confirmed independently by plantarflexor MVC rising
  toward positive angles, passive-torque sign, and EMG/torque sign.
  → plantarflexors are short there (active insufficiency), MVC is 15–28 % of
  peak, retest active-torque std is tiny (YES −0.486: **0.40 Nm**).
* With such a small signal, per-position R² collapses from **offsets and gain
  errors of a few tenths of a Nm** that are harmless at strong angles (std
  8–15 Nm).  Seven concrete mechanisms were measured (below).
* Final pipeline = `python ML/training/train.py --subject <S> --recommended`
  (all **honest** = scored on the original target, no test-session torque
  used):
  1. static-hold passive map (fixes the JM bug),
  2. passive-gain correction `P_eff = α·P + β` fitted on training-session
     rest segments (removes the session offset),
  3. causal EMG adaptive gain control (AGC, `auto`: on unless the training
     session shows pushes in both directions — off for CH),
  4. mixture-of-experts LSTM head (one linear expert per angle, shared trunk)
     trained on per-angle **scale**-normalised targets (each angle weighs the
     same in the loss; no per-angle mean prior, which breaks on CH's task flip).
* Every subject improves at its plantarflexed angles and overall (table at the
  end): YES −0.486 −1.43 → +0.20, JM −0.644 0.02 → 0.80, JM −0.360 0.17 →
  0.93, EG −0.739 0.34 → 0.96, IES −0.591 0.70 → 0.97, HM −0.789 0.67 → 0.92,
  CH −0.737 0.92 → 0.99; pooled MSE lower for all six subjects.
* Trade-off: for YES −0.486 alone, `--emg-agc off` scores higher R² (0.46 vs
  0.20; RMSE 0.29 vs 0.36 Nm — the signal there is only 0.4 Nm std) but
  −0.381 drops to 0.83 (vs 0.87) and pooled MSE rises to 2.09 (vs 1.57).

---

## What causes the drop (measured, not guessed)

| # | Mechanism | Evidence |
|---|-----------|----------|
| M1 | **Session offset of the passive component.** Torque at rest *during active trials* ≈ 0.77·(passive-hold torque) + 0.3 Nm (tissue conditioning / thixotropy), and the hold torque differs between sessions (YES −0.486: 7.9 vs 10.6 Nm) → plain subtraction leaves a session-dependent offset. | YES −0.486: rest-level target −1.32 (test) vs −1.98 Nm (retest); 0.47 Nm bias vs 0.40 Nm std → R² −1.0 alone. α fitted on training session: 0.77 (YES), 1.04 (JM). |
| M2 | Passive torque **relaxes within a trial** (YES −0.486 relaxed plateau sinks ~1 Nm over 90 s; active swing ~1.2 Nm). | plots; rest-level slope 0.53 Nm/90 s at YES −0.486, ≤0.05 sd elsewhere. |
| M3 | **EMG gain changes between sessions**, per angle (JM −0.64 / −0.36: retest plantarflexor envelopes 0.4–0.5× for the same torque; YES −0.38: 1.3×). | decomposition: JM −0.644 prediction slope 0.33 (amplitude under-predicted); YES −0.381 slope 1.33. |
| M4 | Electrode artefact (JM −0.36 retest GL pop ≈ 6× train max) → −11 Nm predicted spike. | clipping alone: JM −0.360 R² 0.41 → 0.49. |
| M5 | High EMG noise floor in JM (rest envelope 15–25 % of max). | — |
| M6 | **MSE ignores weak angles** (YES σ 0.4 vs 15 Nm → ~1400× less loss weight). | even with a clean target the model kept a 0.5 Nm rest bias at YES −0.486. |
| M7 | **JM passive-map bug**: movement segments entered the auto-detected maps after `_PASSIVE_TQ_BOUND` was raised 10 → 20 Nm (for YES, which uses worksheet overrides anyway): test map +11.72 Nm at −0.071 rad (trial 21), retest map +10.75 Nm at −0.064 rad. | JM −0.078 rest-level target −13.6 / −12.2 Nm; static-hold map → −0.3 / −0.5. Most of JM's pooled-MSE gain comes from this fix. |
| — | **CH protocol flip** (found during the regression check): at −0.33 rad CH did a *dorsiflexion* task in the test session (TA, +13 Nm mean) but a *plantarflexion* task in the retest session (GM/GL, −15.6 Nm). | Any model with per-angle priors (mean torque per angle) fails there; models must rely on EMG. |

Not a cause (tested): **position-sensor compliance leak.** The measured angle
deflects 2–3 mrad per 10 Nm (footplate compliance; a linear fit of torque on
the angle wobble explains 93–98 % of torque variance at strong angles, ~0 % at
the weak ones) and a position-only model reaches R² 0.64–0.75 at strong YES
angles — but replacing the angle by its constant set-point leaves strong-angle
R² unchanged for YES, HM and EG, so the LSTM does **not** rely on it.  (For JM
the set-point even helps a bit — its measured angle is noisier.)

Notes vs. the existing notes in the main checkout:
* `subject_preprocessing_summary.md` is outdated: with the segment splitting,
  the pipeline already finds full passive maps for EG, CH (8 pts) and HM (7).
  Every passive trial starts with a clean 20–30 s static hold.
* `YES_passive_torque_issue.md` calls −0.487 "maximum dorsiflexion" — it is
  maximum plantarflexion; the physiological explanation flips accordingly.

---

## Code changes (worktree, uncommitted)

New:
* `ML/preprocessing/active_torque.py` — static-hold passive map; EMG-only rest
  detection; passive-gain fit `P_eff = α·P + β` on training-session rest
  segments; within-trial relaxation baseline (training labels only).
* `ML/preprocessing/emg_agc.py` — causal adaptive gain control of EMG
  envelopes (running p10 floor / p95 level from past samples only, training
  priors at trial start, artefact sample-and-hold).
* `ML/experiments/` — harness, plugins, queue, pipeline runner, reports and all
  result JSONs (`results/`, `results/pipeline/`).

Modified:
* `ML/preprocessing/dataset_builder.py` — `build_dataset(...)` options
  `passive_method` ('static' | 'legacy'), `passive_gain`, `train_label_rezero`,
  `emg_clip`, `emg_floor`, `emg_agc_norm` (False | True | 'dual' | 'auto'),
  `position_feature` ('raw' | 'setpoint'); `build_windows` gained
  `passive_gain`, `label_baseline`, `position_feature`, `extra_columns`.
  Rest statistics / gain fit use the training part only.  Legacy options
  reproduce the old dataset bit-for-bit (verified against the main checkout).
* `ML/models/lstm.py` — `build_model(anchored=..., moe_knots=...,
  moe_expert_bias=...)`; baseline model unchanged (1001 params);
  serialisable custom layers (`.keras` save/load verified).
* `ML/training/__init__.py` — `train_model(position_balance='weights' |
  'standardize' | 'scale', anchored, moe, moe_expert_bias)` (validation
  weighted/standardised the same way); `PositionStandardizer`;
  `evaluate_model` de-standardises; `load_subject` forwards dataset options.
* `ML/training/train.py` — `--recommended` plus individual flags
  `--legacy-passive --no-passive-gain --label-rezero --emg-clip --emg-floor
  --emg-agc {off,on,dual,auto} --position-feature --position-balance
  --anchored --moe --no-expert-bias`.
* `ML/config.py` — `DATA_DIR` honours `EMG_DATA_DIR` (data stays in the main
  checkout).

Defaults in `build_dataset`: static passive map ON, passive gain ON (both are
target fixes), training-label re-zero OFF.  AGC / MoE are enabled by
`--recommended`.  To reproduce the old pipeline exactly:
`--legacy-passive --no-passive-gain`.

```
python ML/training/train.py --subject YES --recommended
python ML/training/train.py --subject JM  --recommended
```

---

## Caveats (from an independent adversarial verification)

* Code path verified clean: replacing retest torque with noise leaves every
  training array and every model input bit-identical; the only retest-torque
  route into the target is the retest passive map; AGC is exactly causal
  (240 corruption cases per subject, zero changes before t0).
* **Development-set numbers.** ~40 configurations were scored on the retest
  session and the final one was picked from them → expect some optimism
  (≈0.05–0.1 R² at the most selected angles).  IES (never used for any
  selection) is the cleanest out-of-sample check.
* The pre-existing envelope extraction (whole-trial demean + zero-phase 2 Hz
  Butterworth) and FIR decimation look ≈0.1–0.5 s into the future — true for
  legacy and new alike; no R² here is real-time achievable as is.
* JM legacy is unstable across seeds (−0.787: 0.61 / 0.89 / −3.93); compare
  against medians.  YES always uses the worksheet passive maps
  (passive_method has no effect for YES; its target is
  tq − (0.779·P_worksheet + 0.306)).
* Pooled R² is not comparable across target definitions (JM's legacy target
  carried a ≈12 Nm offset at −0.078) — compare per-position R² and MSE.
* The screening used the 'fast' protocol (batch 64); see the original-protocol
  confirmation table for batch 8.

## Recommendations for the next data collection

* Passive holds **before and after** each active trial (or at least each
  session) — passive torque relaxes within a trial and is conditioned by the
  contractions (α ≈ 0.77), a single pre-trial hold mis-estimates it.
* A passive hold at **every** operating angle in **both** sessions (YES has
  none at −0.49; JM's −0.07 entry came from a movement trial).
* Keep the task direction identical between sessions at each angle (CH −0.33
  switched from dorsiflexion to plantarflexion).
* A short MVC / calibration contraction at the start of each session would let
  EMG be normalised per session (an alternative to AGC).
* Report RMSE (Nm) alongside per-position R² — at the plantarflexed angles the
  active torque std is 0.4–0.8 Nm, so R² is extremely sensitive.

## Methods tried (screening: harness, fast protocol batch 64, mean of 3 seeds)

"Honest" = scored on the original target, no test-session torque anywhere.
Variants marked NO redefine the test target with in-trial rest torque (useful
as an upper bound only).

### Methods tried — per-position R² at the weak angles (retest, mean of 3 seeds, fast protocol)

`yes*` = honest, uses only past EMG of the current trial (causal AGC).
`NO` = redefines the test target with in-trial rest torque (upper bound only).

```
Method                                             honest YES-0.486 YES-0.381  JM-0.787  JM-0.644  JM-0.360
-----------------------------------------------------------------------------------------------------------
Baseline (original pipeline)                       yes        -1.43     +0.76     -0.81     +0.02     +0.17
Static passive map + passive gain + angle-weighted yes        +0.47     +0.75     +0.89     +0.24     +0.64
Passive fixes + MoE head (per-angle mean prior)    yes        +0.53     +0.85     +0.89     +0.18     +0.59
Passive fixes + EMG AGC + angle-weighted loss      yes*       +0.24     +0.86     +0.87     +0.73     +0.92
Passive fixes + AGC + MoE (per-angle mean prior)   yes*       +0.15     +0.88     +0.88     +0.85     +0.94
Passive fixes + AGC dual view + MoE                yes*       +0.22     +0.86     +0.81     +0.60     +0.48
Passive fixes + AGC dual view + weights            yes*       +0.28     +0.72         —         —         —
Passive fixes + MoE scale-only (no AGC)            yes        +0.46     +0.83     +0.87     +0.22     +0.66
FINAL: passive fixes + AGC + MoE scale-only        yes*       +0.20     +0.87     +0.86     +0.80     +0.93
— screening-only variants (experiment harness) —
Re-zero target on in-trial rest (uses TEST torque) NO         -1.05     +0.58     +0.78     +0.28     +0.73
Re-zero (drift) + angle weights (uses TEST torque) NO         +0.58     +0.58     +0.88     +0.35     +0.73
Re-zero + zero-anchored LSTM (uses TEST torque)    NO         +0.66     +0.70     +0.81     +0.36     +0.55
Set-point angle instead of measured angle          yes        -0.76     +0.80     +0.59     -0.03     +0.61
Clip EMG artefacts only                            yes        -1.30     +0.79     +0.66     -0.04     +0.49
Passive gain + zero-anchored LSTM                  yes        +0.35     +0.71         —         —         —
```

### YES — baseline vs final (real pipeline, fast protocol, mean of 3 seeds; MSE Nm², RMSE Nm)

```
pos | R2 base  R2 new | MSE base MSE new | RMSE b RMSE n
------------------------------------------------------------------
 -0.486 |   -1.43   +0.20 |     0.38    0.13 |   0.62   0.35
 -0.381 |   +0.76   +0.87 |     1.18    0.64 |   1.08   0.80
 -0.278 |   +0.93   +0.99 |     1.58    0.22 |   1.25   0.47
 -0.176 |   +0.98   +0.99 |     1.27    0.72 |   1.12   0.85
 -0.071 |   +0.98   +0.99 |     1.27    0.64 |   1.13   0.80
 +0.031 |   +0.99   +0.99 |     2.03    2.55 |   1.42   1.60
 +0.138 |   +0.97   +0.98 |     6.41    4.52 |   2.53   2.12
 +0.245 |   +0.97   +0.98 |     4.73    3.16 |   2.17   1.77
 pooled |       —       — |     2.36    1.57 |   1.53   1.25
```

### JM — baseline vs final

```
pos | R2 base  R2 new | MSE base MSE new | RMSE b RMSE n
------------------------------------------------------------------
 -0.787 |   -0.81   +0.86 |     0.65    0.05 |   0.64   0.22
 -0.644 |   +0.02   +0.80 |     0.60    0.12 |   0.77   0.35
 -0.504 |   +0.90   +0.88 |     0.32    0.42 |   0.57   0.65
 -0.360 |   +0.17   +0.93 |     1.46    0.13 |   1.20   0.35
 -0.218 |   +0.92   +0.91 |     0.15    0.16 |   0.39   0.40
 -0.078 |   -2.06   +0.87 |     8.13    0.37 |   2.30   0.61
 +0.055 |   +0.71   +0.92 |     2.12    0.51 |   1.40   0.71
 +0.199 |   +0.66   +0.90 |     2.94    0.87 |   1.68   0.92
 pooled |       —       — |     2.05    0.33 |   1.37   0.57
```

### Original training protocol (batch 8, lr 0.003, patience 25), seed 0 — R² baseline → final

```
YES  -0.486: -0.68→+0.17  -0.381: +0.75→+0.88  -0.278: +0.95→+0.99  -0.176: +0.96→+0.99  -0.071: +0.98→+0.99  +0.031: +0.99→+0.98  +0.138: +0.97→+0.98  +0.245: +0.98→+0.98
JM  -0.787: +0.66→+0.88  -0.644: -0.32→+0.84  -0.504: +0.90→+0.88  -0.360: +0.46→+0.93  -0.218: +0.90→+0.90  -0.078: +0.56→+0.84  +0.055: +0.49→+0.93  +0.199: +0.69→+0.85
```

### All six subjects — baseline vs final (IES / EG / CH / HM were never used to choose the method)

FINAL = `--recommended`; AGC auto is OFF for CH (pushes in both directions), ON for the others.

```
subj | weakest angle  R2 base  R2 new | min R2 base min new | pooled MSE →   new
------------------------------------------------------------------------------------
YES  |        -0.486    -1.43   +0.20 |       -1.43   +0.20 |       2.36    1.57
JM   |        -0.787    -0.81   +0.86 |       -2.06   +0.80 |       2.05    0.33
IES  |        -0.591    +0.70   +0.97 |       +0.70   +0.96 |       2.09    0.63
EG   |        -0.739    +0.34   +0.96 |       +0.34   +0.96 |       2.97    1.53
CH   |        -0.737    +0.92   +0.99 |       +0.92   +0.97 |       5.66    3.58
HM   |        -0.789    +0.67   +0.92 |       +0.67   +0.90 |       0.91    0.61
```

Per-seed values, all other variants and the experiment harness results are in
`ML/experiments/results/` (`python -m ML.experiments.report`, `python -m ML.experiments.final_report`).
