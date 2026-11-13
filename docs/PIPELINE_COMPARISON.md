# Pipeline comparison: this branch vs `main`

Written 13 Nov 2026 on branch `cleanup`, from `experimentation-normalization` at `607c98a`, compared with `origin/main` at `7778c39`.
Reproduce the numerical check with `python EXPERIMENTS/analysis/compare_pipelines.py --subject <S>`.

## Recommendation

**Keep both pipelines. Make the experiment pipeline the canonical path for experiments and the dashboard, and merge `main` by combining the five conflicting files rather than picking one side.**

- **Experiment pipeline (canonical).**
  - Code: `ML/research/` (data, passive models, EMG scaling, windows, training) driven by `EXPERIMENTS/pipeline/run.py` and the experiment specs.
  - Every result on the dashboard came from this pipeline, including A0, the default model. Making `main`'s pipeline canonical would mean re-running every experiment.
- **Audited pipeline (from `main`, kept alongside).**
  - Code: `ML/preprocessing/pipeline.py`, `calibration.py` and `trial_manifest.py`, `ML/training/benchmark.py` and `sweep.py`, `ML/inference.py`, `ML/tests/`.
  - It adds trial-identity checks, run packages, an inference API and unit tests. Experiments don't use it today.
  - It stays as `main` has it, with `main`'s unit tests.
- **`EXPERIMENTS/baseline/retest_map_code/`** is a frozen copy of `main`'s `ML/`. Once `main` is merged it duplicates `ML/`. Keep it until the merge is verified, then remove it. The per-run source snapshots inside `run_collections/` stay as evidence.

Why combine instead of choosing a side for the five files:
- `main` changed them only a little: 9–79 lines each.
- This branch changed them a lot: 53–259 lines each. It added things the experiments need, listed below.
- Taking `main`'s side breaks the experiments outright.
- Taking this branch's side drops `main`'s small fixes, and `main`'s new modules depend on some of them.

## Evidence

### 1. With `main`'s `ML/` as is, the experiments can't run

`ML/research` and `ML/experiments` import names that `main`'s `ML/` doesn't have:

| Name | Where it is defined on this branch | On `main` |
| --- | --- | --- |
| `TRIAL_OVERRIDES`, `PASSIVE_TORQUE_OVERRIDES` | `ML/config.py` | gone; `main` reads reviewed overrides from a JSON file (`--overrides`) |
| `SUBJECTS['CH']` | `ML/config.py` | missing (`main` uses five recordings) |
| `N_EMG`; `build_model(..., anchored, moe_knots, moe_expert_bias)` | `ML/models/lstm.py` | missing; `main`'s `build_model` takes `n_units` and `dropout` instead |
| `ML/preprocessing/active_torque.py`, `ML/preprocessing/emg_agc.py` | new on this branch | not on `main` (no conflict: these files survive a merge) |

Result B in the table below is an `ImportError` for every subject.

### 2. With only those names added back, the experiment data is the same

The script builds the default experiment's arrays in three ways:
- A: this checkout's `ML/`
- B: `main`'s `ML/` as is
- C: `main`'s `ML/` plus only the names above

The default experiment is window 20, each session's own relaxed-hold passive table, and EMG divided by the session-1 maximum, with session 1 → session 2. It compares every array (`X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`).

| Subject | B (`main` as is) | C vs A, largest absolute difference |
| --- | --- | --- |
| YES | ImportError | 0 (identical) |
| JM | ImportError | 0 |
| EG | ImportError | 0 |
| IES | ImportError | 0 |
| CH | ImportError | 0 |
| HM | ImportError | 0 |

So `main`'s versions of the shared reading and filtering code (`flb_reader`, `emg_envelope`, `dataset_builder` passive maps) give exactly the same training and test data for the experiments.

**What this does not cover:**
- Training. It covers data preparation only, so `build_model` and the training loop are not exercised.
- `S2toS1`, and settings other than the default.
- `ML/archive/experiments/` (the older harness), whose results are historical.

### 3. Step by step

| Step | Experiment pipeline (`ML/research` + `EXPERIMENTS/pipeline/run.py`) | Audited pipeline (`main`: `ML/preprocessing/pipeline.py`, `ML/training/benchmark.py`) |
| --- | --- | --- |
| Recordings | 6 people (`SUBJECTS` includes CH) | 5 people (no CH) |
| Reading | `flb_reader.read_flb` | same |
| EMG envelope | `emg_envelope.extract_envelope` on the whole trial, then down-sampled to 100 Hz | same function, but each 80/20 block is filtered separately, with a 1 s gap and 1 s trimmed at each edge |
| Passive torque / target | per-session static table from relaxed holds (`passive='static'`, the A0 default); `'none'` = total torque | `measured − fixed passive calibration` from training-session plateaus; `session_passive_calibration` exists but **defaults to False**; `measured_torque` mode for total torque |
| EMG scaling | ÷ session-1 maximum (`trainmax`); p99, MVC and per-session variants | scale fitted on processed training blocks only |
| Window | 20 samples (0.2 s), stride 1 | 50 samples (0.5 s), training stride 5, evaluation stride 1 |
| Train/val split | first 80 % / last 20 % of each training-session trial, no gap | chronological 80/20 blocks with a 1 s boundary exclusion |
| Model | LSTM 8×2, dropout 0.3 (plus GRU, MoE, ridge options) | LSTM, ridge, MLP, position baseline |
| Training | Nadam 0.003, batch 8, ≤200 epochs, ReduceLROnPlateau 0.7/10, early stop 25; per-angle balancing | batch 64, ≤200 epochs, early stopping, standardized torque |
| Scoring | per-position R², MAE, VAF on the test session; dashboard medians | equal-position RMSE per subject, then equal-subject average |
| Outputs | `EXPERIMENTS/results/<exp>/…/seedK.{npz,json}` → dashboard | run packages with hashes, manifest, preprocessing, model |
| Tests | `EXPERIMENTS/tests` (pytest) | `ML/tests` (unittest) |

The two pipelines differ in window, split, filtering and scoring, so their scores can't be compared directly.

## How to resolve each conflicting file

Start from this branch's version of each file and add `main`'s changes on top:

1. **`ML/config.py`.**
   - Keep this branch's `TRIAL_OVERRIDES`, `PASSIVE_TORQUE_OVERRIDES` and the CH entry.
   - Add `main`'s `EMG_DATA_DIR` environment override, `LSTM_UNITS` and `CAPACITY_SWEEP_UNITS`.
   - Keep `DATA_DIR`'s default as `ML/subjects`, where the recordings are tracked; `main` defaults to `ML/data`.
   - **Decision needed:** this branch gives HM and IES the `'IES01'` channel profile; `main` gives them `'default'`. The experiment pipeline produces identical data either way (see the table above), but the older `dataset_builder` path uses the profile. Ask whoever recorded HM which channel order is right.
2. **`ML/models/lstm.py`.** Combine the two signatures into `build_model(n_steps=20, n_features=5, n_units=8, dropout=0.3, anchored=False, moe_knots=None, moe_expert_bias=True)`. Keep `N_EMG` and this branch's MoE/anchored layers, and use `n_units`/`dropout` where `main` does.
3. **`ML/preprocessing/dataset_builder.py`.**
   - Keep this branch's additions.
   - Add `main`'s `passive_entries_override` parameter and its revised std-threshold comment.
   - **Use `env, stages = extract_envelope(...)` and `stages['rectified']`.** `main` changed `extract_envelope` to return a dict, and that file merges without a conflict, so the branch's `env, rect = …` line would break silently.
4. **`ML/training/__init__.py`** and **5. `ML/training/train.py`.** `main` made `train`, `cross_subject` and `fine_tune` run its audited runner. Keep `main`'s entry points. Keep this branch's additions as functions the experiment code calls, and check every caller with `git grep`.

After the merge, check:
- `python EXPERIMENTS/analysis/compare_pipelines.py --ref HEAD~1 --subject <S>`: A must be identical for all six people.
- `python -m unittest discover -s ML/tests`
- `pytest`
- one short training run of `python -m EXPERIMENTS.pipeline.run` (EG, seed 0)

## Decisions for you

1. Do you accept the recommendation: keep both, the experiment pipeline canonical, and combine the five files as above?
2. HM/IES channel profile: `'IES01'` (this branch) or `'default'` (`main`)?
3. Should the audited pipeline's `session_passive_calibration` default become `True`, to match your rule that session-2 passive torque is the default?
