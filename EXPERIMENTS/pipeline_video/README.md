# EMG → ankle torque explainer videos

Two narrated manim videos live here. **v2** (below) explains last semester's baseline in plain English and is the
one to watch. **v1** (further down) walks through the D01 code itself.

## v2: the baseline, in plain English

A 3Blue1Brown-style explainer of the baseline (X00: the frozen working-copy code, LSTM with 8 cells, seeds 0–9,
learn session 1 → test session 2). It covers the inputs and how they are cleaned, cut into windows, split and fed to
the network; training (epochs, early stopping); where the answer key comes from; how a prediction is scored and why a
right-looking prediction can have a negative R²; seeds; overfitting; why toes-down positions score badly; the bugs
found on the way; the potential techniques (auto-gain, equal weight per position, longer windows); what the four
papers did differently; and next steps. No code on screen. Voice: Kokoro-82M `af_sarah`, speed 1.0, 0.6 s between
beats.

**Video:** `renders2/EMG_torque_baseline_explained.mp4`, 1920 × 1080, 30 fps, H.264 + AAC, with chapter markers.
Rendered media and audio are not in git; `build2.py` rebuilds them.

### Chapters

Total 59:02 (1920 × 1080, 30 fps, H.264 + AAC, 187 MB). Times from `renders2/chapters.md`, written by
`build2.py concat`.

| # | Chapter | Starts at | Length |
|---|---|---|---|
| 1 | A puzzle | 0:00 | 1.8 min |
| 2 | The experiment | 1:45 | 3.1 min |
| 3 | What the model must predict | 4:52 | 2.7 min |
| 4 | From raw EMG to a smooth envelope | 7:34 | 2.4 min |
| 5 | One scale per muscle | 9:57 | 1.4 min |
| 6 | Cutting the recording into windows | 11:23 | 1.3 min |
| 7 | Who learns from what | 12:44 | 2.2 min |
| 8 | The network, from far away | 14:55 | 1.7 min |
| 9 | Training: epochs and early stopping | 16:37 | 2.8 min |
| 10 | Seeds: same code, different answers | 19:23 | 3.1 min |
| 11 | Scoring a prediction | 22:30 | 3.2 min |
| 12 | The offset trap | 25:44 | 3.6 min |
| 13 | Overfitting, or a different session? | 29:17 | 3.5 min |
| 14 | Why toes-down is hard | 32:46 | 3.9 min |
| 15 | Bugs and loose ends | 36:42 | 4.1 min |
| 16 | Potential techniques: auto-gain | 40:50 | 4.3 min |
| 17 | Potential techniques: equal weight and longer windows | 45:08 | 4.5 min |
| 18 | What the papers did differently | 49:38 | 5.0 min |
| 19 | Summary and next steps | 54:37 | 4.4 min |

### How it is built

1. `STORYBOARD2.md` is the single source of the narration: 19 chapters, 139 beats, each with `VIS:` (what is on
   screen), `SAY:` (narration) and `SRC:` (where every number comes from).
2. `extract2/*.py` (project interpreter `.venv311`, no training) compute every number and array from the recordings,
   the frozen baseline code and the saved runs, and write `data2/facts_*.json` (in git) and `data2/*.npz` (not in git).
3. `narration2/parse_storyboard.py` turns the storyboard into `narration2/narration.json`; `narration2/tts2.py`
   synthesises one wav per beat (`audio2/`), with the pronunciation pass of v1 (`narration/pronounce.py`) plus a few
   extra rules (run names such as X00f, years, "versus", a comma fix after numbers).
4. `scenes2/cNN.py` (one manim scene per chapter, shared helpers in `scenes2/style2.py`) read only `data2/`, `data/`
   and the narration timings. Each beat plays its wav and lasts its audio plus the gap. `scenes2/preview.py cNN`
   renders a 480p preview with one still per beat; `scenes2/DATA_INDEX.md` lists every array the scenes can use.
5. `build2.py concat` joins the 19 chapter files and writes the chapter markers and `renders2/chapters.md`.

| Script | Produces | Chapters |
|---|---|---|
| `y01_baseline_scores.py` | baseline scores per group in the dashboard's format, offset decomposition, learning-curve summaries, seed spread | 1, 10–13, 19 |
| `y02_inputs.py` | EG inputs rebuilt with the frozen code: envelopes, angle, torque, passive tables, the example window, the fit/validation boundary, where each session-1 maximum lies | 2–7 |
| `y03_predictions.py` | per cell × seed predictions and scores (R², shape-only R², offset, MAE), YES and JM examples | 1, 10–12 |
| `y04_learning.py` | learning curves, learning-rate cuts, stopping epochs of the 60 runs, test ÷ validation ratios | 9, 13 |
| `y05_passive.py` | passive trials and the legacy stillness rules, the JM bug, YES's table vs its relaxed level, loss shares | 3, 10, 14, 15 |
| `y06_techniques.py` | auto-gain, equal weight, longer windows and D01 against the baseline, matched seeds | 16, 17 |
| `y07_reference_checks.py` | "session 1's average" reference, per-trial fit/validation/test R², pooled R², JM against its own targets, dropout loss ratio, CH | 8, 11–15 |
| `c04_trim.py` | EG soleus envelope before the 1 s trim (start and end of the trial) | 4 |

### Review history

- **Understanding:** parallel readers mapped the frozen baseline code, the framework rebuild, the saved runs and the
  four papers before any script was written.
- **Storyboard:** written from those notes, then checked by six independent fact-checkers (every number against code,
  data or paper), a clarity review and a completeness review; a recheck confirmed the fixes. Extra numbers they asked
  for are computed in `y07_reference_checks.py`. Corrections that changed the story are listed below.
- **Scenes:** nine authors (two or three chapters each), each followed by a visual QA pass on the preview stills and a
  fix pass. A final read of every contact sheet against the storyboard found one remaining bug: in chapter 6 the
  torque plot sat about 0.13 s off the signal rows, so the target dot was not under the window's last sample (fixed).
- **Final video:** seven reviewers checked frames of the rendered 1080p file (the middle and the end of every beat)
  against the narration and the facts files: no high-severity issues, 6 medium and about 30 low. All were fixed and
  each group re-verified, except the paper citations on screen (kept on purpose) and a cosmetic offset of the bullet
  dots. Fixes that changed meaning: ch. 5 now shows EG's real session-1 soleus maximum (one sample, in another trial)
  next to the example trial's own peak; ch. 16 draws YES like HM ("barely changes", as narrated); the schematic curves
  in ch. 5 and 12 are labelled as sketches; ch. 14's scatter axes state their direction (session 1 − session 2,
  predicted − true); the envelope is in mV in every chapter; YES's top position reads 0.25 rad (0.2451).

Corrections from the fact-check worth knowing (the video states the corrected version):

- 5 of the 24 session-1 maxima (person × muscle) fall in the validation part, not 6.
- The MVC trials are recorded once at the start and are not used by the baseline.
- Passive trials are relaxed only for their first 20–30 s; gravity is part of the passive torque.
- Scoring uses the clean static-hold table, which differs from the training table only for JM.
- JM scored against its own (buggy) targets: median R² 0.38 over 10 seeds, 3 runs below zero.
- The limit on a passive piece's average is 10 Nm in git and 20 Nm in the working copy; the SD limit is 20 in both.
- The offsets at EG and IES are not explained by the answer key; the model itself reacts to changed session-2 EMG.
- In Nm, the seed-to-seed wobble is about the same everywhere (0.17 vs 0.14 Nm); R² magnifies it at toes-down.
- Toes-down cells score 0.96–0.98 on validation: little to predict amplifies errors rather than causing them.
- Dropout tests: MAE got worse when dropout changed; R² moved within the seed noise.
- Paper details: Zhang 2022 fed 5-step slices and its transfer numbers are worst cases; Kim's zero-torque condition is a
  milder analogue of our problem; Zhang 2019's cut-off is mostly at 1 Hz, its lower bound; Ranaldi's "best" holds only
  by a correlation measure that favours smooth envelopes.

Choices made while building the scenes (all disclosed on screen or in the narration):

- Ch. 4 shows the real envelope before the trim (`c04_trim.py`) rather than a sketch.
- Ch. 11 beat 4 ("same error, different R²") is an illustration: one fixed error series added to two real traces,
  with the resulting R² values (0.999 and −0.57) computed, not typed.
- Ch. 12 beat 4 (the same offset at a strong position) uses a to-scale schematic with a 10 Nm spread, as narrated.
- Ch. 12 beat 5 shows each cell's offset and shape shares of 1 − R² as 100 % bars, since the cells' shortfalls range
  from 0.03 to over 1.5.
- Ch. 16's 32/18/53/30 % size check is quoted from the 27 Oct weekly update (`scripts/agc_summary.py`, last week's
  settings), not recomputed here.

### Rebuild

Same interpreters as v1 (see below). From this folder:

```bash
python build2.py extract
python build2.py narration
python build2.py tts
python build2.py render
python build2.py concat
```

`render` runs `RENDER_JOBS` chapters at a time (default 8; 6 was used, as each 1080p render needs 1–2 GB of RAM) and
deletes manim's partial files after each chapter. `python build2.py render c06 c11` re-renders only those chapters.

### Layout

```
STORYBOARD2.md   the v2 script: VIS / SAY / SRC per beat
build2.py        extract → narration → tts → render → concat
extract2/        data scripts (project interpreter)
narration2/      parse_storyboard.py, narration.json, tts2.py
scenes2/         style2.py, preview.py, DATA_INDEX.md, one scene per chapter
data2/           facts_*.json (+ *.npz, not in git)
audio2/, renders2/  generated, not in git
```

## v1: what the D01 code actually does
A narrated manim video of the D01 pipeline (`D01_cleanpassive+autogain+equalweight`, shown in the dashboard as
**Balanced positions**), from the raw recordings to every R² on the dashboard. Voice: Kokoro-82M, voice `am_adam`.

**Video:** `renders/EMG_torque_pipeline_explained.mp4`, 1920 × 1080, 30 fps, with chapter markers. Rendered media and
audio are not in git; `build.py` rebuilds them.

### Chapters

Total 20:08 (1920 × 1080, 30 fps, H.264 + AAC, 73 MB). Times from `renders/chapters.md`, written by `build.py concat`.

| # | Chapter | Starts at | Length |
|---|---|---|---|
| 1 | The data | 0:00 | 1.9 min |
| 2 | The problem | 1:56 | 1.2 min |
| 3 | The pipeline in one picture | 3:06 | 0.9 min |
| 4 | Preprocessing, step by step | 3:59 | 1.3 min |
| 5 | EMG scaling and auto-gain | 5:15 | 2.7 min |
| 6 | Passive map and target | 7:59 | 1.7 min |
| 7 | Windows and splits | 9:40 | 1.7 min |
| 8 | Balanced positions | 11:23 | 1.5 min |
| 9 | Inside the LSTM | 12:53 | 3.6 min |
| 10 | Prediction | 16:31 | 0.7 min |
| 11 | Evaluation, every number | 17:11 | 1.9 min |
| 12 | Last week's pipeline, and recap | 19:07 | 1.0 min |

### How every number gets on screen

1. `extract/*.py` (project interpreter `.venv311`) import D01's own code from the `dorsiflexion-r2-drop-341137`
   worktree, compute everything from the recordings and saved runs, and write `data/*.npz` (arrays, not in git) and
   `data/facts_*.json` (numbers, in git). `extract/code_excerpts.py` snapshots every code excerpt shown, with the file's
   SHA-256, into `data/code_excerpts.json`, because the D01 code is uncommitted in that worktree.
2. `narration/values.py` turns the facts into display strings. The narration (`narration/narration.json`, one entry per
   beat) and the scenes both use these strings, so a spoken number and the number on screen are the same value.
3. `narration/tts.py` fills the narration, applies the pronunciation pass (`narration/pronounce.py`: E M G, R squared,
   newton metres, radians, G M / G L / soleus / T A, numbers as words) and synthesises one wav per beat with
   `KPipeline(lang_code='a')`, voice `am_adam`, speed 1.1, 24 kHz.
4. `scenes/chNN_*.py` (manim Community 0.22) read only those files. Each beat plays its wav with `add_sound` and is padded
   to the audio's length. `build.py concat` joins the chapters with ffmpeg and adds the chapter markers.

| Script | Produces | Chapters |
|---|---|---|
| `x01_data.py` | trial inventory, raw 1 kHz snippets (both sessions), one active trial, angles, push count | 1, 3 |
| `x02_envelope.py` | one 3 s soleus piece after every preprocessing step, filter gains, look-ahead of the 2 Hz filter | 4 |
| `x03_scaling.py` | ÷ max, push directions, priors, running floor/top, one rescale by hand, despike counts, size ratios, the ÷-max cancellation check | 2, 3, 5 |
| `x04_passive.py` | passive-trial pieces and the stillness test, maps of all six people, targets | 2, 6 |
| `x05_windows.py` | `prepare` + `drop_last_val_window` for EG: one window, shapes, split, knots, SDs | 2, 7, 8 |
| `x06_train_probe.py` | **instrumented rerun of EG seed 0** (≈ 40 min, one thread): the first 5 steps in full, every epoch, all weights | 8, 9 |
| `x07_lstm_window.py` | one window through the trained LSTM in NumPy, gate by gate; the Nadam step; loss shares per angle | 8, 9 |
| `x08_scores.py` | every score rescored from the saved runs; the dashboard's aggregation | 10, 11 |
| `x09_last_week.py` | last week's EG seed-0 settings, predictions and scores | 12 |

### Checks (all passed)

| Check | Result |
|---|---|
| Instrumented rerun vs saved run `results/C01_autogain+equalweight/EG/S1toS2/seed0` | 38 epochs, best epoch 13, best validation loss 0.014504178427159786: all equal; test predictions identical (max difference 0.0 Nm) |
| Probe vs untouched training, one epoch | weights at every recorded step, epoch end and predictions bit-identical |
| NumPy LSTM forward pass vs Keras | 2.5 × 10⁻⁷ Nm |
| Nadam step 1 recomputed from `nadam.py` vs recorded weights | exact for all 1,001 weights |
| Worked auto-gain rescale (JM, session 2, soleus) vs the array value | exact |
| Envelope steps vs `extract_envelope`, and decimation vs the cached bundle | exact |
| EG passive maps rebuilt from the holds vs the cached maps | exact |
| EG arrays built with the C01 config vs the D01 config | identical |
| 624 run × angle scores rescored vs the dashboard's `results.json` | ≤ 5 × 10⁻⁵ (its rounding) |
| Headline and per-person numbers on the built dashboard (`data/check_dashboard_dom.json`) | all equal to `x08_scores.py` |
| Last week's EG seed-0 scores rescored vs the presentation catalog | exact |

### Rebuild

Two interpreters, as for the rest of the project:

- `PROJECT_PY`: `EMG_TQ_Modelling/.venv311` (TensorFlow 2.21) for `extract`.
- `VIDEO_VENV`: `krasi_ml/EMG_TQ_Modelling_video_venv` (Python 3.11 with `manim`, `kokoro`, `soundfile`), created with
  `py -3.11 -m venv` and `pip install --no-cache-dir manim kokoro soundfile`. The Kokoro model (≈ 330 MB) downloads on
  first use into `HF_HOME` (default: `hf_home` inside that venv). espeak-ng is not needed system-wide: `misaki`
  brings `espeakng-loader`.
- ffmpeg on `PATH`. No LaTeX is needed (all text is Pango).
- Data: `EMG_DATA_DIR` = the main checkout's `ML/subjects`; the cached bundles in the dorsiflexion worktree's
  `ML/research/_cache` are what the runs used.

```bash
python EXPERIMENTS/pipeline_video/build.py extract
python EXPERIMENTS/pipeline_video/build.py tts
python EXPERIMENTS/pipeline_video/build.py render
python EXPERIMENTS/pipeline_video/build.py concat
```

`render` runs 8 chapters at a time; each needs about 1.5 GB of RAM. Previews: `manim render -ql ...` from `scenes/`.

### Layout

```
STORYBOARD.md   the approved storyboard, the fact check, and what changed while building
build.py        extract → tts → render → concat
extract/        data scripts (project interpreter)
narration/      narration.json, values.py, pronounce.py, tts.py
scenes/         style.py + one manim scene per chapter
data/           facts_*.json, code_excerpts.json, check_dashboard_dom.json (+ *.npz, not in git)
audio/, renders/  generated, not in git
```
