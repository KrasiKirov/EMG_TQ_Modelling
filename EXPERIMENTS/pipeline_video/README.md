# EMG → ankle torque: what the code actually does (explainer video)

A narrated manim video of the D01 pipeline (`D01_cleanpassive+autogain+equalweight`, shown in the dashboard as
**Balanced positions**), from the raw recordings to every R² on the dashboard. Voice: Kokoro-82M, voice `am_adam`.

**Video:** `renders/EMG_torque_pipeline_explained.mp4`, 1920 × 1080, 30 fps, with chapter markers. Rendered media and
audio are not in git; `build.py` rebuilds them.

## Chapters

<!-- CHAPTERS -->

## How every number gets on screen

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

## Checks (all passed)

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

## Rebuild

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

## Layout

```
STORYBOARD.md   the approved storyboard, the fact check, and what changed while building
build.py        extract → tts → render → concat
extract/        data scripts (project interpreter)
narration/      narration.json, values.py, pronounce.py, tts.py
scenes/         style.py + one manim scene per chapter
data/           facts_*.json, code_excerpts.json, check_dashboard_dom.json (+ *.npz, not in git)
audio/, renders/  generated, not in git
```
