# The current baseline, and how it compares with Zhang 2022 and Truong 2023

Checked against the code and the saved run configs on 13 Nov 2026.

## Short answers

- **Window length.** The default model, **A0**, uses **200 ms** windows: 20 samples at 100 Hz. The 500 ms window is
  a different model, `main/session-passive`. It sits in the same "Models" group of the dashboard but is not the default.
- **Overlap.** Windows overlap heavily, because a new window starts every sample (10 ms). For A0, consecutive windows
  share **95 %** of their samples (19 of 20). For the 500 ms model, training windows share **90 %** (step 5 of 50)
  and test windows **98 %** (step 1 of 50).
- **Session-2 passive torque.** Yes, both models use it. Each session's active trials have that session's own passive
  torque subtracted, so session-2 trials use the passive torque from session 2's relaxed holds. This is part of the
  default, not calibration (see [`CALIBRATION.md`](CALIBRATION.md)).

Overlap is (window − step) / window. A window of W samples moved by S samples shares W − S samples with the next one.

## The two models on the dashboard's "Models" group

| | **A0** (default, landing page) | **`main/session-passive`** (500 ms) |
| --- | --- | --- |
| Code | `ML/research` + `EXPERIMENTS/pipeline/run.py`; spec `EXPERIMENTS/configs/A0_active_trainmax.json` | `main`'s audited pipeline (`ML/preprocessing/pipeline.py`); runs in `EXPERIMENTS/run_collections/window50_retest_map/` |
| Inputs per time step | 4 EMG envelopes (GM, GL, SOL, TA) + ankle angle = 5 features | same 5 features |
| EMG envelope | band-pass 30–300 Hz, rectify, 2 Hz low-pass, at 1 kHz, then down-sampled to 100 Hz | same filter, applied separately to each train/validation block |
| EMG scaling | ÷ the session-1 maximum of each channel | scale fitted on the training blocks only |
| Window | **20 steps × 10 ms = 200 ms** | **50 steps × 10 ms = 500 ms** |
| Step between windows | 1 sample (10 ms) for training, validation and test | 5 samples (50 ms) for training; 1 sample for validation and test |
| Overlap of consecutive windows | **95 %** everywhere | **90 %** training, **98 %** validation and test |
| What is predicted | torque at the **last sample** of the window (estimation, no look-ahead) | same |
| Target | **active torque** = measured − passive torque, with **each session's own relaxed-hold table** (`"passive": "static"`); IES has no passive recording, so nothing is subtracted (total torque); the window study uses IES's MVC-trial passive table instead | **active torque** with `session_passive_calibration = True` in these runs (session-2 passive from session-2 plateaus). The code's default is `False` |
| Train / validation | first 80 % / last 20 % of every session-1 trial | chronological 80/20 blocks with a 1 s gap at the boundary |
| Test | every session-2 trial, whole trial | session-2 trials |
| Model | LSTM 8 × 2 layers, dropout 0.3, per-angle loss scaling (`"balance": "scale"`) | LSTM 8 × 2, dropout 0.3 |
| Training | Nadam 0.003, batch 8, ≤ 200 epochs, LR × 0.7 after 10 flat epochs, early stop 25, windows shuffled | batch 64, ≤ 60 epochs, early stop 12, loss on standardized training torque, windows shuffled |
| Calibration | none | none |
| People, seeds | 6 people, seeds 0–2 | 6 people, seeds 0–9 |

Sources:
- A0: `ML/research/splits.py` (`windows`, stride 1) and `ML/research/config.py` (`WINDOW = 20`, 100 Hz).
- 500 ms model: the `preprocessing.config` of each run's `package.json` (`window = 50`, `train_stride = 5`,
  `target_mode = active_torque`, `session_passive_calibration = True`).

## What the two papers do

**Zhang et al. 2022** ("Lower-limb joint torque prediction using LSTM neural networks and transfer learning", IEEE
TNSRE 30:600–609) and **Truong et al. 2023** ("EMG-based estimation of lower limb joint angles and moments using
LSTM", Sensors 23:3331). The PDFs are in the Capstone folder.

| | **Zhang 2022** | **Truong 2023** | **Ours (A0)** |
| --- | --- | --- | --- |
| Task | 10 dynamic activities (walking at 3 speeds, stairs, squat, sit-to-stand…) | dynamic tasks (squats, picking up, sit-to-stand…), ~5 min continuous | isometric pushes at 8 fixed ankle angles |
| EMG | 13 muscles, 1 kHz, band-pass 30–300 Hz, rectify, **6 Hz** low-pass, ÷ max over all trials | 14 muscles, 1.1 kHz, band-pass 20–450 Hz, 50 Hz notch, differentiated, scaled to [0, 1] | 4 muscles, 1 kHz, band-pass 30–300 Hz, rectify, **2 Hz** low-pass, ÷ session-1 max |
| Features per step | 13 EMG envelopes + 4 joint angles = 17 | AR(6) coefficients + RMS + MAV per channel = 8 × 14 = 112 | 4 EMG envelopes + ankle angle = 5 |
| Feature windows | none; the envelope is used sample by sample | **100 ms windows, 50 ms step → 50 % overlap**, giving a 20 Hz feature stream | none; the envelope is used sample by sample |
| LSTM window | **5 time steps**. The step rate is not stated; at the 100 Hz motion-capture rate this would be about 50 ms | **20 time steps at 20 Hz, called 1 s** (the timestamps span 950 ms; the raw EMG behind them about 1.05 s) | **20 steps at 100 Hz = 200 ms** |
| Step between LSTM windows (overlap) | **not reported** | **not reported** (the 50 % is between feature windows, not between LSTM windows) | 1 sample → **95 %** |
| Prediction time | the current torque (no horizon stated) | the label **one step (50 ms) ahead** | the current torque (last sample of the window) |
| Target | **net joint torques from inverse dynamics** (hip F/E, hip Ab/Ad, knee F/E, ankle D/P), in Nm/kg | **net joint moments from OpenSim inverse dynamics** (+ joint angles), 8 outputs | **active ankle torque** (measured − passive), Nm |
| Model | 2–4 LSTM layers of 50 units, dense 64/128, dropout 0.4, Adam 1e-3, batch 50, ≤ 4000 epochs, early stop 50 | **LSTM 8 × 2, dropout 0.3, Nadam 0.003, LR × 0.7, batch 8, MSE** | the same LSTM 8 × 2 as Truong, same optimizer settings |
| Test data | intra-subject 80/20; leave-one-subject-out; leave-one-task-out; transfer learning | chronological 60/20/20 within one recording | a different session (session 1 → session 2) |
| Shuffling during training | **not reported** | **not reported** | whole windows reshuffled every epoch |

## What this means for our procedure

**Our LSTM is Truong's LSTM.** Truong's architecture and training settings match last semester's LSTM-8, which is
still the A0 network: 2 × 8 units, dropout 0.3, Nadam 0.003, LR factor 0.7, batch 8, MSE. The inputs differ:
- Truong feeds hand-made features from 100 ms windows (AR, RMS, MAV) at 20 Hz, 20 steps = about 1 s of history.
- We feed a smooth envelope at 100 Hz, 20 steps = 0.2 s of history.

So the step count is the same, but Truong's network sees **five times more history at five times coarser time
resolution**. Our 2 Hz envelope filter already carries several hundred milliseconds of past EMG in each sample, so the
real history our network sees is longer than 200 ms. The difference is still large. A 1 s window is the direct way to
test Truong's setting; the earlier 0.4 s (X06) and 0.5 s (D02, and the `main` model) windows sit in between.

**The 50 % overlap in Truong is not the same overlap as ours.**
- Truong's 50 % is how much consecutive 100 ms **feature windows** share (a new feature every 50 ms).
- Our 95 % is how much consecutive **LSTM input windows** share (a new window every 10 ms).
- Neither paper says how far apart its LSTM windows are. If Truong moved one feature step at a time, its LSTM windows
  overlap 95 % (19 of 20), the same as ours.
- In our setup, overlap does not leak test information, because the test is a different session. It does mean
  neighbouring training windows are nearly identical, so the number of windows overstates how much independent data
  there is. Shuffling them changes only the batch order.
- For Truong, high overlap combined with a chronological 60/20/20 split of one recording means windows next to a
  split boundary share data with the other side.

**Truong forecasts 50 ms ahead; we and (as far as stated) Zhang estimate the present.** Predicting ahead is harder.
Truong's headline numbers mix that harder task with an easier test: the same recording, not a new session.

**Zhang uses a much shorter history.** Five steps (about 50 ms if at 100 Hz) with a 6 Hz envelope, but bigger
networks (2–4 layers of 50 units) and 17 inputs including four joint angles that move during the task. In dynamic
movements, joint angles and their changes carry much of the torque. In our isometric trials the angle is constant
within a trial and only says which of the 8 positions it is.

**Both papers predict total (net) torque; we predict active torque.**
- Inverse dynamics gives the **net joint moment**, the total torque about the joint from all muscles, tendons and
  passive tissue. Neither paper removes a passive component.
- In our isometric setup, the measured torque is that net moment, so their target corresponds to our **total
  torque**. That is the N0/N1–N3 normalization study (`"passive": "none"`), not A0.
- A0 subtracts each session's relaxed-hold passive torque and predicts only the voluntary part. That is a deliberate
  difference: the passive part depends on angle and drifts between sessions, and EMG cannot explain it.
- To compare with the papers on their own terms, compare their numbers with N0, not with A0. Even then the tasks
  (dynamic movement vs. isometric), the scales (Nm/kg vs. Nm) and the test designs (same recording or other people
  vs. another session) differ too much to rank the models by headline error.

## Shuffling during training

**What we do.** Both models shuffle during training: Keras `fit(shuffle=True)` reshuffles the order of the
**whole windows** every epoch (`"shuffle": true` in `BASE`, `ML/research/train.py`). Shuffling never touches what is
inside a window:
- the 20 (or 50) samples of each window stay in time order;
- each window keeps its own target, the torque at its last sample.

The LSTM starts every window from a zero state, so it never carried memory from one window to the next anyway.
Shuffling therefore does **not** throw away EMG history: the history a prediction can use is exactly the window.
Validation and test windows are scored in order, and their order does not affect the result.

**Why it matters.** Without shuffling, the windows arrive trial by trial and in time order. Batch after batch comes
from one ankle angle and one stretch of one trial, so each stretch of updates pulls the network toward that angle.
The angles near the end of the epoch dominate the weights. Shuffling mixes angles and trials in every batch.

**What we measured (preliminary).** Turning shuffling off made the model worse in both tests we ran, but neither
test is big enough to settle the question:

| Test | Setup | Shuffled | Not shuffled |
| --- | --- | --- | --- |
| `S0_noshuffle` vs N0 (`EXPERIMENTS/results/`) | 200 ms, total torque, session 1 → 2; the 9 completed runs (CH, EG×2, HM, IES, JM×2, YES×2), each paired with the same N0 person and seed | MAE plantarflexed **1.07** Nm, other angles **1.21** Nm | plantarflexed **1.24** Nm, other angles **1.62** Nm; worse in **9 of 9** pairs |
| JM ablation, 10 Nov (`ML/reports/shuffle_ablation_JM_20261110/`, local, not in git) | 500 ms audited pipeline, JM only, 3 seeds, identical initial weights per pair | retest RMSE **0.55** Nm (validation 0.34) | retest RMSE **0.66** Nm (validation 0.51); worse in **3 of 3** pairs |

Why this is not conclusive:
- `S0_noshuffle` is partial: 9 of the planned 18 runs, one direction, seeds 0–1 only, and it uses total torque (N0),
  not the default A0.
- Its 9 pairs are not 9 independent results, because several come from the same person. Counting people instead,
  unshuffled was worse for 6 of 6 people. That is suggestive (a sign test gives p ≈ 0.03), but 6 people with one or
  two seeds each is a small sample.
- The JM ablation is one person and 3 seeds (3 of 3 favour shuffling, which is p = 0.25 even before asking whether JM
  is typical). Its unshuffled runs mostly hit the 60-epoch cap while still improving, so they may simply have needed
  longer.
- We keep shuffling on because it is the standard default and nothing so far argues against it, not because these
  tests prove it is better. A properly sized test is described below.

**The papers.** Neither Zhang nor Truong says whether training windows were shuffled. Keras shuffles by default, and
Truong's optimizer settings are Keras-style, but that is a guess, not something the paper states. Their results
therefore give no evidence for or against shuffling.

## LSTM state: reset every window, or carried through the trial

**What we do.** Both models are *stateless*: the LSTM's hidden and cell states start at zero for every window.
- The state is updated step by step **inside** a window, over its 20 (or 50) samples.
- It is thrown away at the end, so the next window, which starts 10 ms later, recomputes almost the same history from
  scratch.
- Each prediction therefore depends only on its own window, 200 ms (or 500 ms) of EMG, plus whatever the 2 Hz
  envelope filter has already smoothed in.

The alternative is a *stateful* LSTM: run through the whole trial and pass the final state of one chunk into the next.
The network then has, in principle, everything since the start of the trial.

**What we measured (preliminary, inconclusive).** Three local experiments from 10 Nov tested this, all on the 500 ms
audited pipeline:

| Experiment (`ML/reports/…`) | Comparison | Result |
| --- | --- | --- |
| `state_carry_ablation_JM_20261110` | JM, 3 paired seeds, 500 ms chunks, same starting weights; **reset** each chunk vs **carry** state to the next chunk within a trial | Carry worse: session-2 RMSE **0.630 → 0.845 Nm** (+34 %), R² 0.897 → 0.791; worse in 3 of 3 seeds |
| `state_carry_replication_20261110` | the same test on EG, HM and YES | **Mixed.** Carry better for EG (1.880 → 1.675 Nm) and YES (1.681 → 1.479); about equal for HM (0.910 vs 0.918). Across all four people, mean RMSE slightly favours carry (1.229 vs 1.275), mean R² favours reset (0.941 vs 0.917) |
| `continuous_memory_JM_20261110` | JM, state carried through 2 s chunks for the whole trial vs the normal shuffled 500 ms model | Continuous memory slightly worse on average: session-2 RMSE **0.584 vs 0.546 Nm**, driven by one bad seed (better in 2 of 3). The same continuous weights with state reset every 500 ms scored 0.569 |

**What that means.** These tests **do not answer the question**:
- 4 people with 3 seeds each, and the people disagree: carrying hurt JM, helped EG and YES, and made no difference
  for HM.
- The averages split too: RMSE favours carrying, R² favours resetting.
- 3 seeds per person cannot separate a real per-person effect from seed-to-seed noise. In the JM tests the seeds
  themselves disagree (e.g. continuous memory was better for 2 of 3 seeds and much worse for the third).
- The tests never used the A0 setup.
- Resetting stays the default because it is what every result so far was trained with, not because it has been shown
  to be better.

Resetting also keeps two things the experiment pipeline relies on:
  - training on shuffled windows, which carry-state training cannot do (the windows must stay in order within a
    trial);
  - each prediction depending only on the last few hundred ms, which is simple to run online and cannot drift with
    a state built up over a whole trial.

Caveats:
- In all three tests, training gradients still covered only one chunk (500 ms or 2 s), even when the state was
  carried. Training on whole trials, or with denser targets, could behave differently.
- The two continuous runs that hit the 60-epoch cap may not have finished learning.
- These results are on the 500 ms audited pipeline, not on A0's 200 ms setup, and A0 has not been tested with
  carried state.
- The three report folders are **local only**: `ML/reports/` is ignored by git, so they are not in the repository.

**The papers.** Neither Zhang nor Truong says anything about carrying LSTM state between windows. Both describe the
input as fixed-length sequences: 5 steps in Zhang, 20 steps in Truong. That is the usual stateless setup, the same as
ours, but neither paper states it outright.

## Check of the EMG input pipeline (13 Nov 2026)

What A0 does to each EMG channel (`ML/preprocessing/emg_envelope.py`, `ML/research/data.py`, `ML/research/emg_norm.py`):
1. Subtract the trial mean.
2. Band-pass 30–300 Hz, 4th-order Butterworth, **zero-phase** (`sosfiltfilt`: run forwards and backwards).
3. Rectify.
4. Low-pass 2 Hz, 4th-order Butterworth, zero-phase.
5. Drop the first 1 s of the trial.
6. Down-sample 1 kHz → 100 Hz (zero-phase FIR, `scipy.signal.decimate`).
7. Divide by the largest envelope value of that channel over all session-1 trials (`trainmax`).

All six people, all 384 active-trial channels, were measured.

**Band-pass, rectify, low-pass: correct and standard.** These are the same steps as Zhang (who low-passes at 6 Hz).
The 2 Hz envelope has under 0.03 % of its power above 10 Hz.

**The start of the trial: cutting 1 s is right.** Each envelope was recomputed with 5 s removed from one end and
compared with the uncut envelope. The difference shows how far filter start-up effects reach into a trial:

| Distance from the edge | Start: median / 95th pct error | End: median / 95th pct error |
| --- | --- | --- |
| 0 ms | 84 % / 285 % | 10 % / 96 % |
| 100 ms | 24 % / 76 % | 7 % / 54 % |
| 250 ms | 12 % / 42 % | 1.4 % / 9.6 % |
| 500 ms | 3.3 % / 12 % | 0.4 % / 2.7 % |
| 1 s | 0.2 % / 0.7 % | 0 % / 0.3 % |

(error = absolute difference ÷ the trial's median envelope)

**The end of the trial: it should be cut too.**
- A zero-phase filter has start-up effects at **both** ends.
- At the end the effect is smaller, but the last 250–500 ms are still clearly distorted (95th percentile 10–54 % within
  the last 250 ms).
- A0 keeps those samples as inputs and **scores** them on session 2.
- Cutting the last 1 s, like the first, removes 1.1 % of each 89 s trial.
- `main`'s audited pipeline already trims 1 s at both edges.

**Scaling (÷ session-1 maximum): correct in practice, fragile by construction.**
- The maximum is taken over the whole 1 kHz envelope, **including** the cut first second.
- In 12 of 384 trial-channels the trial's own maximum falls in the first or last second.
- In none of the six people did that change the session-1 maximum actually used (ratio 1.000 for all 24 channels), so
  current results are unaffected.
- The maximum is not driven by single spikes: max ÷ 99.9th percentile has a median of 1.017.
- It is still safer to take the maximum after trimming.
- It also includes the validation part of the session-1 trials. That was the original pipeline's choice and does not
  touch the session-2 test.

**The envelope looks into the future.**
- A zero-phase low-pass puts half its weight on later samples. For the 2 Hz filter, 90 % of that future weight lies
  within **376 ms ahead**.
- So the EMG value at time t already contains up to ~0.4 s of later EMG. The model's "200 ms window" really sees
  roughly 0.2 s + 0.4 s of the past and about 0.4 s of the future.
- This is fine for offline estimation and is what every result so far used. It is **not available in real time**
  (an exoskeleton cannot use future EMG), and it blurs the window-length question.
- A causal (forward-only) envelope would remove the look-ahead but delays the envelope by a few hundred ms.

**Decision (13 Nov 2026): causal envelope from now on.** New experiments (the window study onwards) use
`"envelope": "causal"` with the first and last 1 s cut. Reasons:
- The lab's MATLAB system-identification code, which the project statement compares the ML models against on the same
  datasets, is causal by default (`functions/emg_envelope.m`: `filter_phase` defaults to `'non_zero_phase'`; the
  forward-backward band-pass is commented out).
- The intended uses (prosthetic and orthotic control) can only use past EMG.
- Zhang 2022 and Truong 2023 do not say whether their EMG filters were causal (Zhang states zero-lag filtering only for
  joint angles and torques).

Earlier results (A0 and everything before it) used the zero-phase envelope. They are valid offline estimates but
optimistic for real-time use, and they are not directly comparable with the causal results.

## A test big enough to settle shuffling and state carrying

Both questions need the same kind of test: the A0 setup, every person, enough seeds to see past seed noise, and a
decision rule written down before the results.

| | Choice | Why |
| --- | --- | --- |
| Setup | A0 (200 ms, active torque, session-2 passive), changing only the one thing tested | the results should apply to the default model |
| People | all 6 | the person, not the seed, is the unit that has to replicate |
| Direction | session 1 → 2 (session 2 → 1 as a supplement) | the primary protocol |
| Seeds | 5 per person and condition, paired: same seed and same starting weights in both conditions | 3 seeds could not separate the JM effects from noise |
| Training budget | the A0 limit (≤ 200 epochs, early stop 25), checked that runs are not stopping at the limit | the 60-epoch cap left some 10 Nov runs unfinished |
| Score | per-person median of the per-angle MAE over seeds, plantarflexed and other angles separately, on whole session-2 trials | the dashboard's measure |
| Decision rule (fixed in advance) | a change wins only if it lowers plantarflexed MAE for at least 5 of 6 people without raising the other angles' MAE by more than 0.05 Nm in more than one person; otherwise keep the default | the same style of rule as the normalization study |

Cost:
- **Shuffling off** needs no new code: an A0 copy with `"shuffle": false`.
  - That is 30 new runs (6 people × 5 seeds).
  - A0 itself needs 12 more runs (seeds 3–4).
  - The saved A0 run took 77 min on one thread; the merge check took about 15 min on 8 threads.
- **Carrying state** needs new training code in `ML/research/train.py`: windows kept in trial order and state passed
  between chunks. It is better tested with gradients over whole trials, not one chunk.
  - The runs are 30 per condition, as above.
  - The code needs its own tests (carry = one forward pass; reset = independent windows), as the 10 Nov scripts had.

## Suggested follow-ups (not run)

- A0 with a 1 s window (`"window": 100`), to match Truong's history length.
- A0 with Truong-style 20 Hz steps, i.e. 20 steps of 50 ms: windows from every fifth sample, so the step count stays
  20 but the history is 1 s.
- A0 trained with a training step of 5 samples (90 % overlap, as in the `main` model), to check whether the 95 %
  overlap matters for anything other than run time.
- The shuffling and state-carry tests above.
