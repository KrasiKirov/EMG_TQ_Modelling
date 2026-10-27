# EMG → ankle torque explainer: storyboard and narration

Approved 27 Oct 2026 ("not different days"; the rest left to my judgement). The spoken text that was rendered is
`narration/narration.json`; it follows this storyboard with the changes below, all forced by the extracted data.

**Changes made while building**
- Sessions: "separate blocks of trials in the same file, recorded on the same day" (your answer).
- 6.1: the second part of each passive trial is not the device moving the ankle: the angle steps once and then stays
  constant (SD 0.06 mrad) while the torque swings by tens of Nm. The narration says exactly that.
- 5.7: EG's EMG sizes are not "already close" in general. Over all 8 angles the typical calf mismatch is ×1.38 before
  and ×1.11 after auto-gain (JM: ×1.51 → ×1.08). Only at −0.038 rad are they close. The beat now gives these numbers.
- 5.8: the ÷ session-1 max does not cancel exactly inside auto-gain: the outputs differ on 1.9 % of EG's samples, in
  the 5 trials where the despiker's `lo > 0.1` rule fires (checked: never where it doesn't).
- 1.4: push count is a computed 22 (torque below the midpoint of its 5th–95th percentile for ≥ 0.5 s).
- Added 8.5 (each angle's share of the training loss in the trained model: toes-down 1.5 % plain vs 23.0 % scaled)
  and 9.10 (the rerun is bit-identical to the saved run).
- Voice speed 1.1 (Kokoro `am_adam`) to bring the video to ≈ 20 min; singular "newton metre" after 1 and "per".

Below is the storyboard as approved.

## 0. What the video explains

- **Model:** experiment `D01_cleanpassive+autogain+equalweight`, shown in the dashboard as **Balanced positions**.
  For EG and YES the stored runs are `C01_autogain+equalweight` (config `passive: legacy`). For both people the legacy and
  static passive tables are identical in both sessions (checked on the cached bundles), so those runs are D01's.
- **One job:** `EXPERIMENTS/code/run_experiment.py` → `ML.research.run.prepare` → `drop_last_val_window` →
  `ML.research.train.fit_predict` (all in the `dorsiflexion-r2-drop-341137` worktree, uncommitted there).
- **Dashboard numbers:** `EXPERIMENTS/code/presentation_metrics.py` → `build_dashboard.py` →
  `EXPERIMENTS/dashboard_react/src/data/results.json` (this branch), aggregated in `ExperimentTab.tsx` / `aggregate.ts`.
- **Versions** (from the run JSON): TensorFlow 2.21.0, Keras 3.14.0, NumPy 2.4.4, Python 3.11.1.
- File references below: `ML/research/…`, `ML/preprocessing/…` and `EXPERIMENTS/code/…` are in the dorsiflexion worktree;
  `EXPERIMENTS/dashboard_react/…`, `ML/preprocessing/pipeline.py` and `ML/training/*benchmark.py` are on this branch;
  `keras/src/…` is in `.venv311/Lib/site-packages`. Because the D01 code is uncommitted, the extraction step stores every
  excerpt shown on screen with its SHA-256 (`data/code_excerpts.json`), so the video stays traceable if those files change.

## 1. Fact check: your list against the code

✓ = matches. **≠** = differs. *nuance* = right, but the code does more or less than the sentence says.

| # | You wrote | What the code does | Verdict | Where |
|---|---|---|---|---|
| 1 | 6 people | CH, EG, HM, IES, JM, YES | ✓ | `ML/config.py:18-25` |
| 2 | 2 sessions **on different days** (S1 test, S2 retest) | Both sessions are trials inside **one .flb file per person** (EG: trials 8–23 "test", 24–39 "retest"). The file name carries one date. The trial headers have no clock time (`domainStart` = 0 for every trial). Nothing in the code or data shows different days; my earlier notes say same day, boot re-strapped. | **≠** (unsupported) | `dataset_builder.py:87-100`, `flb_reader.py:209` |
| 3 | isometric trials at fixed angles | 8 angles per session, ankle held still: angle SD 2.0–3.8 mrad within EG's active trials; every active trial 90.1 s raw | ✓ | probe on EG |
| 4 | GM, GL, SOL, TA + angle + torque | File headers "MG EMG", "LG EMG", "Sol EMG", "TA EMG", "Position", "Torque"; columns reordered to [gm, gl, sol, ta]. Raw EMG unit: the reader's docstring says mV, the file doesn't say (EG values ≈ ±0.05) | ✓ (unit unknown) | `emg_envelope.py:26,38-71`, `flb_reader.py:32-35` |
| 5 | 1000 Hz, ↓10 to 100 Hz | Envelope is computed at 1 kHz on the **whole** trial; then the first 1000 samples (1 s) are dropped; then `scipy.signal.decimate(…, 10, ftype='fir', zero_phase=True)` (201-tap anti-alias FIR). Angle and torque are decimated the same way. Trial length after that: 8,910 samples for every trial of every person | *nuance* | `data.py:62-63, 90-104` |
| 6 | demean → band-pass → rectify → low-pass → downsample | Same order. Band-pass 30–300 Hz and low-pass 2 Hz, both 4th-order Butterworth applied with `sosfiltfilt` (forward + backward: zero phase), then clip at 0. **Non-causal:** 90 % of the 2 Hz filter's weight lies within ±0.38 s, 99 % within ±0.87 s. So `emg_agc.py`'s "a real-time decoder can compute exactly the same features" holds for the auto-gain step only, not for the envelopes it receives | ✓ + *nuance* | `emg_envelope.py:120-144` |
| 7 | ÷ the session-1 training maximum | ÷ per-channel max of the **1 kHz** envelope over **whole** S1 active trials: training *and* validation parts, plus the first second that is trimmed afterwards. S2 is divided by the same numbers. For the 5 people with auto-gain on, this division cancels in the auto-gain output except through two absolute constants (despiker's `lo > 0.1`, the 1e-3 range floor) | *nuance* | `data.py:102`, `emg_norm.py:85-91`, `emg_agc.py:69` |
| 8 | floor p10 / 20 s, top p95 / 45 s, every 0.5 s; priors n/(n+500); min range 0.35 × prior range; clip [0, 2]; despiked | All ✓. Details: percentiles use past samples only (`x[u−win:u]`) and are held for 0.5 s. n = samples in that window (≤ 2000 floor, ≤ 4500 top). Priors = median over S1 trials of each trial's p10 / p95 on its **training part**. Every trial restarts from the priors, S2 trials included. Despike runs first (> 4 ranges above floor, or < 0.25 × floor when floor > 0.1 → hold last good value 0.3 s), then floor/top are recomputed on the cleaned signal. Output = clip((x − floor) / max(top − floor, rmin), 0, 2) | ✓ | `emg_agc.py:25-105`, `emg_norm.py:60-70,104-108` |
| 9 | off for CH | Rule `agc_auto`: off when one training trial's push direction is > +2 Nm and another's < −2 Nm (torque in top-20 % EMG minus bottom-20 % EMG). CH S1: +18.0, −54.8, +16.5, −31.6, +28.0, +9.4, −45.1, −54.5 Nm → off (also S2→S1). On for EG, HM, IES, JM, YES | ✓ | `emg_norm.py:96-103,141-151`, run JSONs |
| 10 | relaxed holds ≥ 5 s, angle SD < 5 mrad, torque SD < 1 Nm → mean = one map point per session | Thresholds ✓ (≤, on 1 kHz samples). Holds are pieces of trials whose comment says "passive", cut at device steps (> 5 rad/s) with 1 s skipped after each step. **"Relaxed" is never checked with EMG.** Holds within 0.05 rad are averaged into one point; < 2 points → fallback to the old rule. YES: hand-entered table in `ML/config.py` for both sessions. **IES has no passive trials → empty map → nothing subtracted.** JM and HM: 7 points per session (JM's movement recording is rejected). EG: 8 holds of 20.1 s per session | ✓ + *nuance* | `dataset_builder.py:141-176`, `active_torque.py:54-80`, `data.py:130-150` |
| 11 | target = measured − passive(angle); S1 map → training target, S2 map → scoring target | ✓. The map is read at **every sample's measured angle** (linear interpolation, linear extrapolation outside). Each trial uses its own session's map | ✓ | `run.py:70-71`, `passive.py:45-49,100-102`, `dataset_builder.py:236-275` |
| 12 | 20 × 5 windows, target at last step, stride 1 | ✓ everywhere (train, val, test); 8,891 windows per trial | ✓ | `splits.py:35-41` |
| 13 | first 80 % train, last 20 % validation, no gap, 19 overlapping windows; last week 1 s gap | ✓ 7,112 / 1,779 windows per trial, then `run_experiment.py` **drops the last validation window of every trial** (→ 1,778, "to match the original split"). Inputs of the first 19 validation windows overlap training windows; their targets do not. Last week: `gap_s = 1.0` (0.5 s each side of the cut) and each block filtered on its own | ✓ + *nuance* | `splits.py:66-72`, `run_experiment.py:65-74`, `pipeline.py:75-85` |
| 14 | test = every S2 window; `shuffle=True` only reorders training windows; S2toS1 swapped | ✓ (71,128 test windows for EG; shuffle = `tf.random.shuffle` of the training indices each epoch). S2→S1 has seeds 0–2 only; S1→S2 has 0–9 | ✓ | `train.py:178-181`, `array_data_adapter.py:133-134` |
| 15 | knots = training angles merged within 0.02 rad | ✓ (angles rounded to 3 decimals, chained merge, knot = mean); EG: 8 knots, none merged | ✓ | `train.py:28-39` |
| 16 | sd per knot = std of that angle's training target, min 0.1 | ✓, population SD of the **training part only**; validation targets are divided by the same values. The dashboard says "each angle scaled by its **swing**": in the code it is the standard deviation | ✓ (label ≠ code) | `train.py:150` |
| 17 | train on y/sd, prediction × sd of nearest knot | ✓ nearest knot to the test trial's mean angle; no sample weights | ✓ | `train.py:151-152,183-184` |
| 18 | MSE weights each angle by SD² (1 − R²) → Σ(1 − R²) | ✓ per angle Σ(y − ŷ)² = N·SD²·(1 − R²) (training windows, angle's own mean); after scaling N·(1 − R²); N = 7,112 for every angle of every person, so the loss is the mean over angles of (1 − R²) | ✓ | derivation |
| 19 | LSTM(8) → Dropout(0.3) → LSTM(8) → Dropout(0.3) → Dense(1) | ✓, 1,001 parameters, built by `run_experiment.py:47-62` (model name `lstm`), not by `ML/models/lstm.py`. Keras defaults: sigmoid gates, Glorot-uniform kernels, orthogonal recurrent weights, forget bias 1 | ✓ | `run_experiment.py:47-62` |
| 20 | Nadam 0.003, batch 8, RLROP 0.7/10, ES 25 + restore, ≤ 200 epochs | ✓ plus: β₁ 0.9, β₂ 0.999, ε 1e-7; loss MSE; RLROP min_lr 1e-6 and **min_delta 1e-4**; ES min_delta 0; 7,112 updates per epoch. Keras 3.14 restores the best weights at the end of training **even when all 200 epochs run** | ✓ + *nuance* | `ML/research/config.py:38-41`, `train.py:173-176`, `early_stopping.py:133-140` |
| 21 | `set_random_seed` → initial weights, shuffle, dropout | ✓ seeds Python, NumPy, TF. Initializer seeds and both Dropout seed generators come from Python's `random`; the shuffle is TF's `tf.random.shuffle`. No `enable_op_determinism` in D01 (last week has it), yet the saved EG learning-curve rerun reproduced 38 epochs / best 13 | ✓ | `train.py:136`, `seed_generator.py:145-146` |
| 22 | per person × angle × seed: R², MAE and **peak error** | R² = 1 − SS_res/SS_tot with SS_tot about the **trial's own mean** ✓; MAE ✓. **The dashboard's "peak" is the peak target torque**, max \|active torque\| of that test trial, shown next to the MAE; it is not an error. Also RMSE, bias, demeaned R²/MAE. Scored at every window's last sample (8,891 per trial) | **≠** (peak) | `presentation_metrics.py:81-88,107-112` |
| 23 | active or total torque? | **Active:** key = measured − the tested session's static map (IES: measured torque, nothing subtracted) | answered | `presentation_metrics.py:70-78` |
| 24 | negative R² shown as 0 | ✓ per run, before any median (toggle, on by default) | ✓ | `metrics.ts:28-35` |
| 25 | medians per position, then across positions, 5–95 % bands | Centre ✓. **The bands depend on the view:** box plots = 5–95 % over seeds at each position; Overview KPI cards = percentiles over seeds of each seed's median across the group's person-position cells; the Balanced-positions headline has no band. Toes-down = each person's 2 most negative test angles | ✓ + *nuance* | `ExperimentTab.tsx:29-34`, `aggregate.ts:34-57`, `data.ts:91-94` |
| 26 | anchor EG −0.04 rad, seed 0, S1→S2: R² 0.986, MAE 0.84 Nm (last week 0.983, 1.08) | ✓ R² 0.9862, MAE 0.8394 Nm at −0.038 rad (results.json, C01 row); last week R² 0.9830, MAE 1.0827 Nm (presentation catalog, `main/session-passive`) | ✓ | `results.json` |

Two more things the code does that no label says:

- `prepare()` always detects EMG rest periods and builds rest flags (`run.py:64-72`), and calls `PassiveModel.fit`.
  With `passive = static` the fit returns at once (`passive.py:144-145`) and D01 has no `rest_weight`, so none of that
  changes D01's data.
- In the dashboard, EG and YES get their D01 numbers from the C01 folders through the "equivalent passive table" rule
  (switch on by default, `ExperimentTab.tsx:53-58`, `params.ts:147-155`).

## 2. Production decisions

- **Length:** ≈ 3,000 narrated words (counts per chapter below) → about 18–20 min. Exact length is known after the voice test.
- **Format:** 1920 × 1080, 30 fps, background `#0E1116`.
- **No LaTeX** on this PC → `Text` / `MarkupText` (Pango) with Unicode maths (R², Σ, ŷ, σ, ⊙, √). Fonts: Segoe UI (text), Consolas (code). Nothing installed system-wide.
- **Palette**, fixed for the whole video:

  | Role | Colour |
  |---|---|
  | GM | `#2EC4B6` teal |
  | GL | `#A78BFA` violet |
  | SOL | `#FFC857` amber |
  | TA | `#F15BB5` magenta |
  | ankle angle | `#C9D1D9` light grey, dashed |
  | session 1 (training) | `#4C9AFF` blue; validation part = same blue, hatched |
  | session 2 (test) | `#FF7A45` orange (same sides as the deck's pipeline tree) |
  | measured / target torque | `#FFFFFF` |
  | passive torque | `#8B949E` grey |
  | prediction | `#9BE564` green |

- **Layout:** chapter tag top-left ("4 · Preprocessing"); animation on the left ~60 %; code panel on the right ~40 % with
  `file:line` in its header. The line that produces the current picture is highlighted while that picture is on screen.
- **Numbers:** scenes never compute anything. They read `data/*.npz` and `data/facts.json`, written by the `extract/`
  scripts, which import the D01 code from the dorsiflexion worktree and run in `.venv311`. Narration text holds
  `{key}` placeholders filled from `facts.json`, so the spoken and the shown number are the same number.
- **Voice:** Kokoro `KPipeline(lang_code='a')`, voice `am_adam`, 24 kHz, one wav per beat. **Sync:** each beat calls
  `add_sound(beat.wav)` and its animations are padded to the wav's duration; the chapters are joined with ffmpeg concat.
- **Pronunciation pass** (applied to the narration text before TTS):

  | Written | Spoken |
  |---|---|
  | EMG, LSTM | "E M G", "L S T M" |
  | R² | "R squared" |
  | Nm, rad, mrad | "newton metres", "radians", "milliradians" |
  | GM, GL, TA | "G M", "G L", "T A" |
  | SOL | "soleus" |
  | CH, EG, HM, IES, JM, YES | "C H", "E G", "H M", "I E S", "J M", "Y E S" |
  | D01, C01 | "D zero one", "C zero one" |
  | Hz, kHz, % | "hertz", "kilohertz", "percent" |
  | −0.038 | "minus zero point zero three eight" |
  | 56,896 | "fifty-six thousand eight hundred and ninety-six" |
  | Nadam | "N-Adam" |

- **Video venv:** `C:/Users/dramo/Desktop/Projects/krasi_ml/EMG_TQ_Modelling_video_venv` (outside git, next to the data and
  runs folders), built from the Python 3.11 in the py launcher. `.venv311` is not touched.
- **Folder:** `EXPERIMENTS/pipeline_video/` with `extract/`, `scenes/`, `narration/`, `audio/`, `renders/`, `data/` and a README.
  `.gitignore`: `audio/`, `renders/`, `media/`, `data/*.npz`.
- **Git:** `experimentation` is checked out in the `open-website-b5b382` worktree, so it can't be checked out here as well.
  This worktree's branch is now fast-forwarded to `experimentation`. Commits go on this branch, and `experimentation` is
  fast-forwarded to them with `git merge --ff-only`, run in the open-website worktree. Never pushed.

## 3. Chapters

| # | Chapter | Words | ≈ min |
|---|---|---|---|
| 1 | The data | 290 | 1:55 |
| 2 | The problem | 180 | 1:10 |
| 3 | The pipeline in one picture | 165 | 1:05 |
| 4 | Preprocessing, step by step | 205 | 1:20 |
| 5 | EMG scaling and auto-gain | 380 | 2:25 |
| 6 | Passive map and target | 270 | 1:45 |
| 7 | Windows and splits | 285 | 1:50 |
| 8 | Balanced positions | 220 | 1:25 |
| 9 | Inside the LSTM | 540 | 3:25 |
| 10 | Prediction | 75 | 0:45 |
| 11 | Evaluation, every number | 290 | 1:55 |
| 12 | Last week's pipeline, and recap | 140 | 0:55 |
| | **Total** | **≈ 3,040** | **≈ 20** |

Balanced positions gets its own chapter (8), between the splits and the LSTM, because the LSTM chapter's batch and loss
already use y / sd.

## 4. Beats

How to read the numbers in the narration: plain numbers are **verified now** (source given). `⟨≈x⟩` = provisional
value from an earlier analysis, recomputed by the extraction script. `⟨?⟩` = computed in step 3. In the narration
file every number becomes a `{key}` read from `facts.json`.

### Chapter 1 · The data

**1.0 · Title** · ~9 s
> EMG to ankle torque: what the code actually does. We follow experiment D01, the model the dashboard calls Balanced positions, from the raw recordings to every R² on the dashboard.

- Screen: title; the six subject codes as badges; small label "D01 · Balanced positions".

**1.1 · Six people** · ~15 s
> There are six people, known here only by their codes: CH, EG, HM, IES, JM and YES. Each has one recording file. We follow EG all the way through, and bring in the others where they behave differently.

- Screen: six badges, EG lights up; "1 recording file each".
- Code: `ML/research/data.py:161-162` (`fn, sid = SUBJECTS[subject]` / `trials = read_flb(...)`).

**1.2 · Inside one file** · ~22 s
> EG's file holds 40 trials in recording order. First, eight maximal contractions of 30 seconds, which this model never uses. Then session one, which the lab calls test: at each of eight ankle angles, a passive trial followed by an active trial. Then session two, the retest: the same eight angles in the same order.

- Screen: a 40-block timeline. MVC blocks grey, session-1 blocks blue, session-2 blocks orange; passive = outline, active = filled; each pair labelled p1…p8 with its angle in rad.
- Code: `ML/preprocessing/dataset_builder.py:87-100` (session and trial type parsed from the comment, e.g. "active test, p1").
- Numbers: 40 trials; 8 MVC × 30.1 s; passive/active 90.1 s; angle order p1…p8 = −0.458, +0.247, −0.599, −0.178, −0.318, −0.739, −0.038, +0.104 rad ← `x01_data.py` (`read_flb` + `_parse_comment` on EG).

**1.3 · Channels** · ~18 s
> Every trial stores six channels at 1000 samples per second: four surface EMG signals, from the medial and lateral gastrocnemius, called GM and GL, the soleus, and the tibialis anterior, TA, plus the ankle angle in rad and the torque in Nm.

- Screen: 2 s of the six raw channels from EG's session-1 trial at −0.038 rad, around a push onset; each in its colour, with units ("EMG: recorded units", "rad", "Nm").
- Code: `ML/preprocessing/emg_envelope.py:26` (`CANONICAL_EMG_ORDER = ['gm', 'gl', 'sol', 'ta']`).
- Numbers: raw samples ← `x01_data.py`.

**1.4 · One active trial** · ~18 s
> Here is one active trial: EG at −0.038 rad. The dynamometer holds the ankle still; the angle's standard deviation is only 2 mrad. For 90 seconds, EG pushes about 20 times. Pushing the toes down makes the torque negative.

- Screen: the full 90 s torque trace (white) with the angle strip (grey dashed); labels "SD 2.0 mrad", "toes-down push → negative torque".
- Numbers: angle SD 2.0 mrad; torque min −32.6 Nm; push count ← `x01_data.py` (EG trial 21).

**1.5 · Eight angles** · ~17 s
> The eight angles run from −0.74 rad, toes down, to +0.25 rad, toes up: negative means plantarflexed. Pushes are small at the toes-down end, reaching about 4 Nm, and about 40 Nm at the toes-up end.

- Screen: an ankle sketch rotating through the eight angles; beside it a bar per angle with the trial's largest |torque| (4.3 → 40.4 Nm).
- Numbers: S1 angles; min torque −4.27 Nm at −0.739 rad, −40.38 Nm at +0.247 rad ← `x01_data.py`.

**1.6 · Two sessions** · ~14 s
> Session one and session two are separate blocks of trials in the same file. In between, electrodes and tissue can change. The real question is whether a model trained on session one still works on session two.

- Screen: the timeline again, session 1 marked "train", session 2 marked "test".

### Chapter 2 · The problem

**2.1 · The task** · ~14 s
> The task: from the last 0.2 seconds of muscle signals and ankle angle, predict the torque the muscles are producing now. Train on session one; score on session two, which the model never sees.

- Screen: a 0.2 s strip of four envelopes + angle → "model" box → one torque value; S1 = train, S2 = test badges.

**2.2 · EMG size changes** · ~16 s
> Three things make this hard. First, EMG size changes between sessions. For JM at −0.36 rad, the session-two soleus signal per Nm of push is only ⟨≈0.38⟩ times its session-one size.

- Screen: JM soleus envelope, session 1 vs session 2 (session colours), with torque underneath; the ratio for GM, GL, SOL.
- Numbers: S2/S1 "EMG size per Nm of push" = (p95 − p10 of the envelope) / (p95 − p5 of the torque), GM ⟨≈0.36⟩, GL ⟨≈0.83⟩, SOL ⟨≈0.38⟩ ← `x03_scaling.py` (same definition as `new_method_data.py`).

**2.3 · Passive torque** · ~17 s
> Second, passive torque. Even with relaxed muscles, stretched tissue pulls on the ankle: about +2.5 Nm at EG's most toes-down angle, −10 Nm at the toes-up end, and the values differ between sessions. The EMG cannot see any of it.

- Screen: EG's passive maps, session 1 (blue) and session 2 (orange); the gap at +0.25 rad marked (−10.14 vs −8.32 Nm).
- Numbers: S1 map +2.54 Nm at −0.738 rad, −10.14 Nm at +0.249 rad; S2 +2.62, −8.32 Nm ← bundle maps (`x04_passive.py`).

**2.4 · Small pushes** · ~21 s
> Third, the toes-down pushes are small. At −0.74 rad, EG's active torque has a standard deviation of 2 Nm; at +0.10 rad, almost 13. One Nm of error is minor at one end and half the signal at the other, and a plain loss barely notices the small angles.

- Screen: active-torque traces at −0.739 and +0.104 rad on the same Nm axis; SD bars 2.02 vs 12.92 Nm.
- Numbers: `sd_per_knot` 2.02 and 12.92 Nm ← `results/C01…/EG/S1toS2/seed0.json`.

### Chapter 3 · The pipeline in one picture

**3.1 · Session 1 branch** · ~26 s
> Here is the whole pipeline for EG. The top branch uses session one only. The raw EMG becomes a smooth envelope, is divided by the session-one maximum, and is auto-gained. A passive map from relaxed holds is subtracted from the measured torque, giving the active-torque target. 0.2-second windows train a small LSTM, with every angle scaled to count the same.

- Screen: the pipeline tree redrawn in manim (same structure as the deck's `pipeline_tree_new.png`, with the video palette): each box lights up as it is named, each with a mini-plot of real EG data.

**3.2 · Session 2 branch** · ~15 s
> The bottom branch is session two. The same preprocessing runs with session one's settings, the frozen model predicts, and the prediction is scored against session two's own active torque, built from session two's own passive map.

**3.3 · Where this lives in the code** · ~22 s
> In the code, one job is one call of run experiment: prepare builds the arrays, fit predict trains and predicts. Prepare also detects rest periods and fits a passive model, but with the static passive setting that fit returns at once, so those steps change nothing for D01.

- Code: `EXPERIMENTS/code/run_experiment.py:117-123`; `ML/research/run.py:62-73`; `ML/research/passive.py:144-145` (`if self.kind in ('legacy', 'static'): return self`).
- Screen footnote: "EG and YES: stored as C01, identical to D01 (same passive tables)" ← `presentation_metrics.py:56`.

### Chapter 4 · Preprocessing, step by step

All six beats transform the **same** real 3 s piece of EG's soleus (session 1, −0.038 rad, around a push onset).

**4.1 · Demean** · ~12 s
> The preprocessing, on one real three-second piece of EG's soleus signal. First, the mean of the whole trial is subtracted: an offset of ⟨≈−0.019⟩ recorded units.

- Code: `ML/preprocessing/emg_envelope.py:122-124`.
- Numbers: mean of the whole 90 s soleus channel ← `x02_envelope.py`.

**4.2 · Band-pass** · ~14 s
> Then a fourth-order Butterworth band-pass, 30 to 300 Hz, removes drift and movement artefacts below 30 Hz and noise above 300. It runs forward and then backward, so it adds no delay.

- Screen: waveform before → after; inset with the filter's gain vs frequency (computed from the same coefficients, applied twice).
- Code: `emg_envelope.py:126-130`.

**4.3 · Rectify** · ~5 s
> Rectification takes the absolute value, so every oscillation becomes a positive bump.

- Code: `emg_envelope.py:132-133`.

**4.4 · Low-pass** · ~20 s
> A 2 Hz low-pass, also run forward and backward, smooths that into the envelope: the size of the muscle's activity. Running backward means using the future: 90 % of this filter's weight lies within 0.38 seconds on either side, so the envelope, as coded, could not be computed in real time.

- Screen: envelope drawn over the rectified signal; then the filter's two-sided impulse response with the ±0.38 s band shaded.
- Code: `emg_envelope.py:135-140`.
- Numbers: ±0.376 s (90 %), ±0.874 s (99 %) ← `x02_envelope.py` (impulse through `sosfiltfilt`).

**4.5 · Trim and downsample** · ~16 s
> The first second of every trial is dropped. Then the envelope is decimated by ten with an anti-aliasing filter: 1000 samples per second become 100, one value every 10 ms. Angle and torque go through the same decimation.

- Screen: the 1 kHz curve becomes dots every 10 samples; "1 s trimmed" shaded at the trial start.
- Code: `ML/research/data.py:62-63` (`decimate(..., ftype='fir', zero_phase=True)`), `data.py:94-98`.

**4.6 · Result** · ~10 s
> Each active trial is now 8,910 samples long: 89.1 seconds of four envelopes, the angle and the torque.

- Screen: array shapes `env_raw (8910, 4)`, `pos (8910,)`, `tq (8910,)`.
- Numbers: trial length ← cached bundles (all 96 active trials: 8,910).

### Chapter 5 · EMG scaling and auto-gain

**5.1 · Step one: ÷ the session-1 maximum** · ~18 s
> Scaling has two steps. Step one divides each muscle by its largest envelope value in session one, taken on the 1 kHz envelope over whole trials, including the parts later used for validation. Session two is divided by the same session-one numbers.

- Screen: EG's four maxima as numbers (recorded units), then the session-1 envelopes scaled to 0…1.
- Code: `ML/research/emg_norm.py:85-91`; `ML/research/data.py:102` (`'env_max1k': env1k.max(axis=0)`).
- Numbers: `train_max` per channel ← `x03_scaling.py` (`normalise` info).

**5.2 · Who gets auto-gain** · ~24 s
> Step two, auto-gain, is switched on only for people who push in one direction. For each session-one trial, the code compares torque during the top fifth of muscle activity with torque during the bottom fifth. EG's eight differences are all negative, from −4 to −24 Nm, so auto-gain is on. CH pushes both ways, from +28 to −55 Nm, so for CH it is off.

- Screen: eight bars per person, EG all below zero, CH on both sides of the ±2 Nm lines.
- Code: `emg_norm.py:141-151` (`push_directions`), `emg_norm.py:96-103` (the rule).
- Numbers: EG −8.43, −19.31, −7.54, −16.98, −12.82, −4.16, −21.86, −24.40 Nm; CH +18.04, −54.79, +16.49, −31.59, +27.95, +9.37, −45.13, −54.47 Nm ← run JSONs (`details.push_directions`).

**5.3 · Running floor and top** · ~19 s
> Auto-gain rescales each muscle from its own recent past. Every half second the code looks back: the floor is the 10th percentile of the last 20 seconds, the top is the 95th percentile of the last 45 seconds. Only past samples are used, so this step is causal.

- Screen: JM session-2 soleus at −0.36 rad; two look-back windows (20 s, 45 s) slide along; floor and top draw as step curves updated every 0.5 s.
- Code: `ML/preprocessing/emg_agc.py:25-33` (constants), `emg_agc.py:36-51` (`causal_quantiles`).

**5.4 · Priors** · ~18 s
> Early in a trial there is little past, so each value is blended with a prior from session one's training data. The data's weight is n over n plus 500, where n is the number of past samples; after 5 seconds data and prior count equally.

- Screen: the weight curve n/(n + 500) over the first 20 s of the trial; floor and top moving from their priors to the data.
- Code: `emg_agc.py:47-49`, `emg_agc.py:86-91` (`agc_priors`), `emg_norm.py:106`.
- Numbers: JM priors per muscle ← `x03_scaling.py`.

**5.5 · One rescale, worked through** · ~30 s
> One rescale, worked through, for JM's session-two soleus. At ⟨?⟩ s, the floor window holds ⟨?⟩ samples, so the data weight is ⟨?⟩. The floor is ⟨?⟩, the top ⟨?⟩, the range ⟨?⟩, above the minimum of 0.35 times the prior range, ⟨?⟩. The value ⟨?⟩ becomes value minus floor, divided by range: ⟨?⟩. Results are clipped between 0 and 2.

- Screen: the formula filled in term by term; the sample marked on the trace.
- Code: `ML/research/emg_norm.py:60-70` (`_agc`).
- Numbers: ← `x03_scaling.py`, recomputing `_agc` at one sample and checking it equals the array value.

**5.6 · Despike** · ~9 s
> Before rescaling, spikes more than four ranges above the floor are held at the last good value for 0.3 seconds. ⟨?⟩

- Code: `emg_agc.py:66-83`.
- Numbers: number of replaced samples for EG and JM ← `x03_scaling.py` (narration says the count, or "none in EG's trials").

**5.7 · Before vs after** · ~20 s
> For JM at −0.36 rad, session two's size per Nm, relative to session one, goes from ⟨≈0.36⟩ to ⟨≈0.91⟩ for GM, ⟨≈0.83⟩ to ⟨≈1.04⟩ for GL, and ⟨≈0.38⟩ to ⟨≈0.98⟩ for the soleus. For EG the sizes were already close, so auto-gain changes little.

- Screen: paired bars before/after for JM; small EG inset (sizes ⟨≈0.93–1.21⟩ before).
- Numbers: ← `x03_scaling.py`.

**5.8 · What step one still does** · ~12 s
> A consequence that only the code shows: auto-gain divides by a range measured on the same signal, so for the five people with auto-gain on, step one's division cancels out, except in one fixed threshold of the spike detector.

- Code: `emg_agc.py:69` (`(lo > 0.1)`).
- Numbers: checked numerically in `x03_scaling.py` (auto-gain of EG with and without the ÷ max: max difference printed).

### Chapter 6 · Passive map and target

**6.1 · A passive trial** · ~14 s
> Now passive torque. Before each active trial comes a passive trial: the ankle is held still at the coming angle while the person relaxes, for about 20 seconds, and then the device moves the ankle.

- Screen: EG's passive trial before p7 (trial 20): angle and torque, still part then movement.
- Numbers: ← `x04_passive.py`.

**6.2 · Finding the holds** · ~25 s
> The code cuts each passive trial wherever the angle jumps faster than 5 rad per second, skips one second after each jump, and keeps a piece only if it lasts at least 5 seconds, its angle varies by at most 5 mrad and its torque by at most 1 Nm. Nothing checks the EMG: relaxed means a still piece of a trial labelled passive.

- Screen: trial 20 cut into pieces; each piece with its length, angle SD and torque SD, kept ✓ or dropped ✗.
- Code: `ML/preprocessing/dataset_builder.py:156-176` (`_split_passive_segments`), `ML/preprocessing/active_torque.py:66-71`.
- Numbers: per-piece stats ← `x04_passive.py`.

**6.3 · The map** · ~25 s
> Each piece that passes gives one map point: its mean angle and its mean torque. Points within 0.05 rad are averaged. For EG that is eight points per session, from eight 20-second holds. So the map is not an average over every moment at that angle; the active trials contribute nothing. Session two gets its own map from its own holds.

- Screen: hold → dot; the eight blue dots (S1) and eight orange dots (S2) joined by straight lines.
- Code: `active_torque.py:71-80`.
- Numbers: EG maps (8 points each, listed in §1) ← bundle maps.

**6.4 · Three exceptions** · ~18 s
> Three exceptions. For YES, the map is a hand-entered table in the config file. IES has no passive trials, so nothing is subtracted and the target is the measured torque. And one of JM's passive trials contains movement, which the stillness test throws out.

- Code: `ML/config.py:47-49` (header of `PASSIVE_TORQUE_OVERRIDES`, not the table's source), `ML/research/data.py:147-149`, `ML/research/passive.py:47-48` (empty map → zeros).

**6.5 · The target** · ~21 s
> The target is measured torque minus the map, read at every sample's angle by straight-line interpolation. For EG at −0.038 rad the session-one map gives about ⟨≈−3.49⟩ Nm. Training targets use the session-one map; the scoring target uses the session-two map, about ⟨≈−2.60⟩ Nm here.

- Screen: measured torque (white), passive (grey) under it, their difference = active target; then the same for session 2.
- Code: `ML/research/run.py:70-71`, `ML/research/passive.py:100-109`, `dataset_builder.py:252-275`.
- Numbers: map values at the trial's angles (median) ← `x04_passive.py`.

### Chapter 7 · Windows and splits

**7.1 · The sliding window** · ~20 s
> The model never sees a whole trial. It sees windows: 20 samples in a row, 0.2 seconds, of five inputs: the four scaled envelopes and the ankle angle in rad. The target is the active torque at the window's last sample. Then the window moves on by one sample, 10 ms.

- Screen: EG session-1 trial at −0.038 rad as five stacked traces; a 20 × 5 frame slides along; beside it the frame's 100 real numbers as a heat grid; the target dot at its last step.
- Code: `ML/research/splits.py:35-41` (`windows`).
- Numbers: one window's values ← `x05_windows.py`.

**7.2 · How many** · ~7 s
> 8,910 samples give 8,891 windows per trial.

- Screen: 8910 − 20 + 1 = 8891.

**7.3 · The 80/20 split** · ~19 s
> Session one is split inside every trial. The first 80 % of windows, 7,112, are for training. The rest are validation, minus the very last window, which the experiment script drops to match the original code: 1,778.

- Code: `splits.py:66-70`, `EXPERIMENTS/code/run_experiment.py:65-74` (`drop_last_val_window`).

**7.4 · No gap** · ~20 s
> There is no gap between the parts. The first validation window shares 19 of its 20 input samples with training windows, and the first 19 validation windows all overlap training data. Their targets are new; their inputs are not. Last week's pipeline left a one-second gap and filtered each part separately.

- Screen: zoom on the boundary; overlapping samples highlighted; last week's layout below it with the 1 s gap.
- Code: `splits.py:68-70` beside `ML/preprocessing/pipeline.py:81-84`.

**7.5 · Shapes** · ~18 s
> Across EG's eight session-one trials, that makes 56,896 training windows and 14,224 validation windows, each 20 by 5. The test set is every window of all eight session-two trials: 71,128.

- Screen: `X_train (56896, 20, 5)`, `X_val (14224, 20, 5)`, `X_test (71128, 20, 5)`.
- Numbers: ← run JSON `n_windows`, confirmed by `x05_windows.py` (calls `prepare` + `drop_last_val_window`).

**7.6 · What shuffle does** · ~25 s
> Shuffle equals true means one thing: at the start of every epoch, Keras shuffles the order of the training windows, mixing trials and angles, and cuts them into batches of 8. Validation windows are never trained on; they are only measured, to decide when to slow down and when to stop. Session two plays no part in training. The reverse direction is the same with the sessions swapped, run with three seeds instead of ten.

- Screen: cards coloured by angle shuffle into batches of 8; validation cards stay in place.
- Code: `ML/research/train.py:178-181`, `keras/src/trainers/data_adapters/array_data_adapter.py:133-134`.

### Chapter 8 · Balanced positions

**8.1 · Knots** · ~13 s
> Now the change behind the name balanced positions. Training angles are grouped into knots; angles closer than 0.02 rad would be merged. EG has eight knots, one per angle.

- Code: `ML/research/train.py:28-39`.
- Numbers: knots −0.739, −0.599, −0.458, −0.317, −0.178, −0.038, +0.104, +0.247 rad ← run JSON.

**8.2 · One standard deviation per angle** · ~25 s
> For each knot, the code takes the standard deviation of the training targets at that angle, with a floor of 0.1 Nm. For EG that runs from 2.02 Nm at the toes-down end to 12.92 Nm at +0.10 rad. The dashboard calls this each angle's swing; in the code it is the standard deviation, from the training part only.

- Screen: eight bars of SD per angle.
- Code: `train.py:150`.
- Numbers: 2.02, 3.65, 4.25, 6.41, 8.05, 10.91, 12.92, 11.31 Nm ← run JSON `sd_per_knot`.

**8.3 · Divide, then multiply back** · ~12 s
> The model trains on each target divided by its angle's standard deviation; its output is multiplied back by the standard deviation of the nearest knot, which gives Nm again.

- Screen: targets at −0.739 and +0.104 rad before (very different heights) and after division (similar heights).
- Code: `train.py:151-152`, `train.py:183-184`.

**8.4 · Why** · ~31 s
> Why it matters: for one angle, the sum of squared errors equals the number of windows, times the standard deviation squared, times one minus R². So under plain mean squared error, EG's +0.10 rad counts 41 times more than −0.74 rad for the same R². After dividing, every angle has 7,112 windows and the loss is the sum of one minus R² over the angles: each angle counts the same.

- Screen: Σ(y − ŷ)² = N · SD² · (1 − R²) built term by term; bars of SD² per angle (relative weight) flatten to equal bars; then each angle's share of the loss for the trained model, plain vs scaled.
- Numbers: (12.92 / 2.02)² = 40.9 ← run JSON; loss shares ← `x05_windows.py` with the probe model's weights.

### Chapter 9 · Inside the LSTM

**9.1 · The model** · ~19 s
> The model is small. An LSTM layer with eight units that passes on its output at every step, 30 % dropout, a second eight-unit LSTM that passes on only its last output, dropout again, and one linear output: 1,001 numbers in total.

- Screen: layer diagram with parameter counts 448 = 4 × 8 × (5 + 8 + 1), 544 = 4 × 8 × (8 + 8 + 1), 9 = 8 + 1.
- Code: `EXPERIMENTS/code/run_experiment.py:54-61` (the builder that made D01's model).

**9.2 · The cell** · ~25 s
> At every step, the input and the layer's previous output are combined into four groups of eight numbers. The input, forget and output gates go through a sigmoid, between 0 and 1; the candidate goes through tanh. New memory equals old memory times the forget gate, plus candidate times the input gate. The output is tanh of the memory, times the output gate.

- Screen: z = x·W + h·U + b → split into i, f, g, o; c = f ⊙ c₋₁ + i ⊙ g; h = o ⊙ tanh(c).
- Code: `keras/src/layers/rnn/lstm.py:277-286` and `lstm.py:221-228`.

**9.3 · One window, step one, real numbers** · ~28 s
> One real window: EG, session two, −0.038 rad, ⟨?⟩, through the trained model. At step one the inputs are these five numbers. For unit ⟨?⟩ of layer one, the input gate sums to ⟨?⟩, which the sigmoid turns into ⟨?⟩. The forget gate is ⟨?⟩, the candidate ⟨?⟩, the output gate ⟨?⟩. The memory starts at zero, so the new memory is ⟨?⟩ and the output ⟨?⟩.

- Screen: the five inputs; the 5 input weights + bias of that unit's four gates, multiplied and summed on screen.
- Numbers: trained weights from the instrumented run (§5), forward pass in NumPy ← `x07_lstm_window.py`, checked against `model.predict` (difference printed).

**9.4 · Twenty steps** · ~24 s
> The same arithmetic repeats 20 times, feeding each output back in. These are layer one's eight outputs and eight memories across the window. Layer two keeps only its last output, eight numbers, and the dense layer turns them into one: ⟨?⟩. Times this angle's standard deviation, 10.91 Nm, that is ⟨?⟩ Nm; the measured active torque was ⟨?⟩ Nm.

- Screen: 8 × 20 heat strips for h and c; layer-2 output (8 numbers); dense; × 10.91 Nm.
- Code: `ML/research/train.py:182-184`.

**9.5 · The seed** · ~20 s
> Training starts from the seed. Set random seed seeds Python, NumPy and TensorFlow. Python's generator fixes the starting weights and the two dropout layers' masks; TensorFlow's seed fixes the shuffle. Layer one's input weights start uniform within ±0.40, recurrent weights start orthogonal, forget-gate biases at one.

- Code: `ML/research/train.py:136`, `keras/src/random/seed_generator.py:145-146`.
- Numbers: Glorot limit √(6 / (5 + 32)) = 0.403; initial weights ← instrumented run.

**9.6 · Step one: a real batch of 8** · ~17 s
> Step one. The first shuffled batch holds eight windows from ⟨?⟩ different angles. Each target is divided by its angle's standard deviation. With dropout on, the model predicts these eight values, and the loss is the mean of the eight squared differences: ⟨?⟩.

- Screen: eight rows: trial, angle, target (Nm), SD, target / SD, prediction, squared error; their mean = loss.
- Code: `keras/src/backend/tensorflow/trainer.py:57-68` (forward pass with `training=True`, loss).
- Numbers: ← instrumented run (`x06_train_probe.py`).

**9.7 · One Nadam update** · ~30 s
> Back-propagation gives a gradient for every one of the 1,001 weights. Nadam keeps running means of the gradients and of their squares, corrects both for the first steps, adds a look-ahead on the first, and divides one by the square root of the other. On step one that moves every weight by almost exactly 0.0032, whatever its gradient. This weight, ⟨?⟩, has gradient ⟨?⟩ and moves from ⟨?⟩ to ⟨?⟩.

- Screen: m, v, m̂, v̂ and the step for one weight, filled with its numbers; then a histogram of all 1,001 first steps clustering at ±0.0032.
- Code: `keras/src/optimizers/nadam.py:114-146` (+ `:95-105` for the momentum product).
- Numbers: ← instrumented run; 0.003 × 1.0628 = 0.00319 from the formula, confirmed on the recorded step.

**9.8 · The first five steps** · ~9 s
> Here are the first five steps: the batch loss, and three weights moving. One epoch is 7,112 such steps.

- Screen: small table and sparkline per weight.
- Numbers: ← instrumented run.

**9.9 · Epochs, LR drops, early stopping** · ~35 s
> After each epoch Keras measures the validation loss with dropout off; the training loss it reports is an average over the epoch with dropout on, so the two curves sit at different levels. EG's best validation loss came at epoch 13: 0.0145. Ten epochs without an improvement of at least 0.0001 cut the learning rate by 30 %, after epoch 23 and again after epoch 33. Twenty-five epochs without any improvement stopped training after epoch 38, and the weights from epoch 13 were put back. Those weights predict session two.

- Screen: train and validation loss per epoch; LR as a step line (0.003 → 0.0021 → 0.00147); marker at epoch 13; "restore" arrow back to it.
- Code: `ML/research/train.py:174-176`, `keras/src/callbacks/early_stopping.py:123-140`.
- Numbers: verified now from the saved learning-curve rerun (`EXPERIMENTS/results/learning_curves/D01…/EG/S1toS2/seed0.json`: best 13, val 0.01450, LR cuts after 23 and 33, 38 epochs) and the run JSON (38 epochs, best 13, best val 0.014504); the instrumented run must reproduce them.

### Chapter 10 · Prediction

**10.1 · One pass** · ~15 s
> Prediction is one pass over all 71,128 session-two windows, with dropout off. Each output is multiplied by the standard deviation of the knot nearest to its trial's angle, which turns it back into Nm.

- Code: `ML/research/train.py:182-184`.

**10.2 · Whole trials** · ~30 s (12 s narration + 18 s of trials playing)
> Here is every session-two trial of EG with seed 0: measured active torque in white, the prediction in green, with each trial's R² and mean absolute error.

- Screen: eight 90 s panels (toes-down at the top), measured vs predicted, R² and MAE in each panel corner.
- Numbers: saved predictions `results/C01…/EG/S1toS2/seed0.npz` + the scoring key ← `x08_scores.py`.

### Chapter 11 · Evaluation, every number

**11.1 · What is compared** · ~18 s
> The dashboard's numbers come from the presentation metrics script. It compares the prediction with active torque, measured torque minus the tested session's passive map, at every window's last sample. Not total torque. For IES, with no map, that is the measured torque itself.

- Code: `EXPERIMENTS/code/presentation_metrics.py:70-78`, `:107-112`.

**11.2 · R², live** · ~28 s
> One angle, computed live: EG, −0.038 rad, seed 0, 8,891 samples. Add the squared errors: the residual sum, ⟨?⟩ Nm². Add the squared distances from this trial's own mean: the total sum, ⟨?⟩ Nm². R² is one minus their ratio: 0.986. Zero would mean no better than a flat line at the trial's own mean.

- Screen: squared errors stacking into a bar (SS_res) beside the squared deviations bar (SS_tot); the ratio; R².
- Code: `presentation_metrics.py:81-88` (`metrics`).
- Numbers: R² 0.9862 ✓ (results.json); sums ← `x08_scores.py`.

**11.3 · MAE and peak** · ~15 s
> The mean absolute error is 0.84 Nm. The dashboard shows it next to the peak target, the largest absolute active torque in the trial: 27.6 Nm. That is a peak torque, not a peak error.

- Numbers: MAE 0.8394 Nm, peak 27.59 Nm ✓ (results.json).

**11.4 · EG's table** · ~14 s
> Every angle and every seed fills EG's table: eight angles by ten seeds, eighty values. Negative R² is set to zero first; then each angle takes the median over its ten seeds.

- Screen: an 8 × 10 table filling cell by cell, then a median column.
- Code: `EXPERIMENTS/dashboard_react/src/lib/metrics.ts:32-35`.
- Numbers: 80 R² values ← `results.json`, cross-checked by rescoring the saved .npz files in `x08_scores.py`.

**11.5 · Across people** · ~20 s
> Across people, each person-and-angle cell gets one vote. The headline R² is the median of all 48 cell medians: ⟨≈0.978⟩; the mean absolute error the same way, ⟨≈0.57⟩ Nm. The toes-down group, each person's two most negative angles, gives ⟨≈0.937⟩; the other 36 cells ⟨≈0.981⟩.

- Screen: six small tables collapse into 48 medians, then into one number; toes-down cells highlighted.
- Code: `EXPERIMENTS/dashboard_react/src/tabs/ExperimentTab.tsx:29-34` (`score`).
- Numbers: my reimplementation of `score()` on results.json (S1→S2: 0.9781 / 0.5746 Nm / 0.9366 / 0.9808); `x08_scores.py` also reads the same numbers off the built dashboard (`data-testid="single-r2"` etc.) and stops if they differ.

**11.6 · The bands** · ~16 s
> In the box plots, the whiskers are the 5th and 95th percentiles over the ten seeds at each angle; with ten seeds, the 5th percentile sits just above the worst seed. On the overview cards the range is taken differently: the median over cells within each seed, then percentiles across seeds.

- Code: `stats.ts:3-11` (numpy-style linear percentile), `aggregate.ts:46-57` (`groupSummary`).

### Chapter 12 · Last week's pipeline, and recap

**12.1 · The contrast** · ~33 s
> Last week's pipeline, for contrast. It used half-second windows and every fifth training window, cut a one-second gap at the split and filtered each part separately, divided the EMG by its training maximum without auto-gain, and standardised the target once for all angles. It trained in batches of 64, for at most 60 epochs with patience 12. On EG at −0.038 rad, seed 0, it scored R² 0.983 and a mean absolute error of 1.08 Nm, against 0.986 and 0.84 Nm for D01.

- Screen: two-column table (last week | D01) with the code line for each row; the two EG predictions overlaid on the same trial.
- Code: `ML/preprocessing/pipeline.py:17-30` (`PipelineConfig`), `pipeline.py:75-85`, `ML/training/benchmark.py:69-78`, `ML/training/parallel_benchmark.py:28-29`.
- Numbers: R² 0.9830 / MAE 1.0827 Nm (presentation catalog), 0.9862 / 0.8394 Nm (results.json) ✓; last week's prediction ← `EMG_TQ_Modelling_runs/window50_retest_map/EG_seed0`.

**12.2 · Recap** · ~20 s
> In one breath: turn the EMG into smooth envelopes, rescale them from their own recent past, subtract a passive map measured in still, relaxed holds, show a small LSTM 0.2-second windows with every angle weighted equally, and score each person and angle on a session it never saw.

- Screen: the pipeline tree from chapter 3, each box ticking off as it is named.

## 5. Data sources (step 3)

All scripts in `EXPERIMENTS/pipeline_video/extract/`, run with `.venv311`, D01 code imported from the dorsiflexion
worktree, data from `EMG_DATA_DIR`. Each writes an `.npz` in `data/` and adds its numbers to `data/facts.json`.

| Script | Produces | For |
|---|---|---|
| `code_excerpts.py` | every code excerpt shown (path, lines, text, SHA-256) | all chapters |
| `x01_data.py` | trial inventory (6 people), EG raw snippets, one active trial, angle SDs, torque extremes | 1 |
| `x02_envelope.py` | the 3 s soleus piece after each step, filter responses, impulse-response extent | 4 |
| `x03_scaling.py` | training maxima, push directions, priors, running floor/top (EG, JM), the worked rescale, despike counts, size ratios before/after, the ÷-max cancellation check | 2, 5 |
| `x04_passive.py` | passive-trial pieces and their stats, holds, maps (6 people), targets | 2, 6 |
| `x05_windows.py` | `prepare` + `drop_last_val_window` for EG S1→S2: one window, shapes, boundary overlap, knots, SD, loss shares | 7, 8 |
| `x06_train_probe.py` | **instrumented rerun of EG seed 0** with the D01 settings (one single-thread job, ≈ 70 min): initial weights, the first 5 batches (inputs, targets, predictions, loss, gradients, Nadam m/v, weights), per-epoch history, restored weights, test predictions | 8, 9 |
| `x07_lstm_window.py` | NumPy forward pass of one window with gate values, from the probe's weights | 9 |
| `x08_scores.py` | per-trial scores from the saved runs, EG's 8 × 10 table, the dashboard aggregation, check against results.json and the built dashboard | 10, 11 |
| `x09_last_week.py` | last week's settings and EG seed-0 predictions and scores | 12 |

**How the instrumented run proves it is the saved run.** It builds the data and model with the same calls as
`run_experiment.py`, sets the same seed and threads (1), and replaces only Keras' `train_step` with an identical
copy that also stores the batch, the gradients and the Nadam state for the first 5 steps. It passes if (1) epochs,
best epoch and best validation loss equal the run JSON (38, 13, 0.014504), (2) the per-epoch history equals the
learning-curve rerun, and (3) its test predictions equal the saved `seed0.npz` (max difference reported). If the copied
`train_step` breaks bit-equality, a plain rerun with a read-only callback provides the history and weights, and the
probe is only accepted if its weights after step 5 equal that run's.

## 6. Questions for you

1. **Sessions:** the narration says "separate blocks of trials in the same file". Were they really on different days?
2. **Raw EMG unit:** the reader's docstring says mV, the file doesn't. "Recorded units" on screen unless you know the amplifier scaling.
3. **SOL:** spoken as "soleus" (GM/GL/TA as letters). Fine?
4. **Git:** OK to commit on this worktree's branch and fast-forward `experimentation` to it in the open-website worktree?
5. **Downloads for step 4:** PyPI packages (manim, kokoro, soundfile and their dependencies, including CPU PyTorch,
   roughly 300 MB) and the Kokoro model from Hugging Face (`hexgrad/Kokoro-82M`: weights ≈ 330 MB, `am_adam` voice < 1 MB).
   espeak-ng is not installed; Kokoro's English front end uses it only for words missing from its dictionary, and a pip
   package can provide it inside the venv. I'll check that during the voice test, without any system-wide install.
