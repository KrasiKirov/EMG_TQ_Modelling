# Implementation plan: reliable EMG-to-torque modelling with existing data

Status: proposed implementation, not executed. Prepared 2026-09-25.

## Scope and decisions

- Use only the five existing FLB recordings: HM, EG, IES, JM, YES. No new recordings, participants, or external training datasets are required.
- Retain the current 2-layer, 8-unit LSTM as the reference architecture.
- Preserve existing checkpoints, plots, and unrelated working-tree changes. All new experiments write to uniquely named run directories.
- Audit existing records and the available IES document/YES worksheet. This is clarification of existing data, not data collection. Where records cannot resolve an ambiguity, retain that uncertainty and use the fallback below.
- Do not promise an accuracy increase from corrections. An initially lower score can be evidence of a more honest evaluation.
- Previously inspected retest results cannot become a pristine test set again. Freeze the revised protocol before new comparisons and describe the final results as retrospective internal validation.

## Ordered implementation milestones

### M0 — Capture the reference and establish a runnable environment

Estimated effort: 0.5–1 engineering day, excluding long training runs.

Files: `requirements.txt`, a new environment lock file, `training/run_artifacts.py` (new), `config.py`.

1. Record the current source revision, working-tree diff, hashes of input FLB files, and hashes of existing checkpoints. A Git commit alone is insufficient because the current worktree contains substantial uncommitted changes.
2. Establish a project environment with compatible Python/TensorFlow versions; freeze resolved versions. The Python used during the review lacked TensorFlow.
3. Run the existing tests and load each saved subject checkpoint. Record architecture and input shape.
4. Evaluate checkpoints with the current preprocessing where compatible. Mark results as reconstructed: saved checkpoints do not establish which historical preprocessing options were used.
5. Save new outputs under `runs/<run_id>/`, including configuration, provenance, timings, history, metrics, and predictions. Never overwrite the existing reference artifacts.
6. Add a `seed` option. Use seed 11 for initial smoke checks, then seeds 11, 23, 37 for screening, and 11, 23, 37, 53, 71 for final shortlisted neural configurations. Record deterministic-operation settings and hardware.

Acceptance: the project can load a model and run inference in the locked environment; a reference report distinguishes reproduced results from historical plots whose configuration is unknown.

### M1 — Inventory the existing trials, with an explicit uncertainty fallback

Estimated effort: 1–2 days. Do not delay later engineering indefinitely while trying to infer undocumented labels.

Files: `preprocessing/flb_reader.py`, `preprocessing/trial_manifest.py` (new), `preprocessing/audit_trials.py` (new), `data/trial_manifest.json` (new).

1. Generate a manifest entry for every raw trial with source hash, subject, trial number, original comment, proposed type/session/position, valid sample intervals, confidence, evidence, inclusion status, and exclusion reason.
2. Read the existing IES metadata document and YES worksheet. Check suspected YES label inconsistencies, particularly trials 27, 29, 31, and 33. Their signal statistics are a reason to investigate, not proof of the correct labels.
3. Inspect channel headers, sampling intervals, record lengths, finite values, position stability, torque variability, EMG activity, clipping, and suspicious discontinuities. Preserve the raw sampling interval instead of silently replacing it with 1 ms; make any justified override explicit.
4. If the original MATLAB reader is already available locally, compare representative parsed records. Otherwise test supported binary layouts with synthetic fixtures and explicitly record that real-file parity remains unverified.
5. Apply verified manifest overrides before classifying trials. Replace the torque-SD-only definition of an isometric trial with acquisition labels plus position-stability checks. Use EMG and torque statistics as quality flags.
6. Generate subject × session × position coverage tables showing both included and excluded trials.

Fallback when metadata is insufficient:

- Mark the trial `ambiguous`; do not relabel it merely because that increases model accuracy.
- Exclude ambiguous records from the primary benchmark and passive calibration. Keep a full exclusion ledger.
- If multiple interpretations are defensible, register them as separate sensitivity analyses before training. Do not select the interpretation with the best retest score.
- Missing positions stay missing. Comparisons use identical eligible trials and also report coverage. If a subject lacks evaluable records for a protocol, mark it unavailable rather than silently substituting another protocol.

Acceptance: every existing trial has a manifest entry; every included interval has documented provenance; the pipeline runs even if some labels remain unresolved.

### M2 — Make target definitions and passive calibration explicit

Estimated effort: 1–2 days; depends on M1's first manifest version.

Files: `preprocessing/calibration.py` (new), `preprocessing/dataset_builder.py`, `training/__init__.py`, `training/passive_torque_diagnostics.py`.

1. Introduce an explicit `target_mode`: `measured_torque` or `active_torque`. Preserve measured torque alongside every corrected target.
2. Keep active-torque modelling as the original scientific objective where passive calibration is supportable. Add a separate measured-torque benchmark on all eligible subjects to enable comparison without fabricating passive measurements. Never pool the two target modes into one score.
3. For IES, missing passive data must produce `active_torque unavailable`, unless the existing metadata provides a justified correction. It must not silently become a zero-passive estimate labelled active torque.
4. Use only manifest-approved passive intervals. Identify stable plateaus using position stability, torque dispersion, and evidence of relaxation; inspect their distributions before fixing quality thresholds.
5. Estimate each plateau robustly and retain dispersion, duration, session, and trial provenance. Do not rely on the current 20 Nm SD / ±10 Nm mean thresholds as sufficient validation.
6. Review the suspicious JM segment currently accepted near −0.065 rad, with mean torque about +7.8 Nm and SD about 17.6 Nm. Include it only if the audit supports it.
7. Default to fixed calibration from permitted source/training-session records. Add a separately labelled session-calibrated protocol only when the existing data contain appropriate calibration records available before the evaluated active segment.
8. Use transparent interpolation between supported positions. Flag extrapolation and unsupported ranges; exclude unsupported active-torque evaluations from the primary score and show them separately in sensitivity analysis. Do not manufacture coverage with an unconstrained curve.
9. Save the entire calibration object with the model, including MVC maps when used. Reject missing/invalid MVC scales rather than silently changing target units.

Acceptance: all target values have an explicit unit and mode; no disallowed retest measurement affects fitted training calibration; missing support is visible in metrics and coverage reports.

### M3 — Isolate splits and preserve provenance through preprocessing

Estimated effort: 2–3 days; depends on M1–M2 interfaces.

Files: `preprocessing/splits.py` (new), `preprocessing/emg_envelope.py`, `preprocessing/dataset_builder.py`, `training/__init__.py`, `training/fine_tune.py`, `config.py`.

1. Replace the 11-element dataset tuple with a structured dataset bundle. Each split carries features, targets, measured torque, MVC scales if applicable, subject/trial/position IDs, timestamps, and raw sample ranges.
2. Separate preprocessing `fit` from `transform`; fit channel scales and other learned transforms on training data or explicitly permitted calibration records only. Validation/test data must not influence fitted parameters.
3. Freeze split membership before filtering, scaling, and window construction. Filter offline temporal blocks independently; trim filter edges on both ends so zero-phase filtering cannot reach across split boundaries.
4. Prefer held-out repeated trials where they exist. Otherwise use blocked validation within each training-session trial so each represented position retains training data. Record that this estimates within-recording performance, not a new-session effect.
5. For blocked validation, start with a one-second exclusion interval around the boundary. Finalize trimming and gap from filter-response/edge-sensitivity checks on training data; one second is a starting value, not a guarantee. Construct windows independently and assert raw-index disjointness.
6. Replace the random overlapping-window split used in fine-tuning with the same blocked procedure. Shuffling training windows after split assignment remains permitted.
7. Derive position groups for weighting/calibration selection from training metadata, not retest operating positions. Use explicit position IDs for evaluation instead of assigning every point to its nearest retest angle.
8. Make window length, stride, sampling rate, and feature layout flow from one configuration into both data construction and model creation. Remove duplicated assumptions that prevent reliable window-length experiments.
9. Make repeated preprocessing calls safe: avoid mutating raw frames and prevent generated envelope columns from being redetected as raw EMG channels.

Acceptance: tests prove zero shared raw samples between train/validation, changing held-out values cannot change fitted transforms, feature order is stable, and changing window length updates model input shape correctly.

### M4 — Build one evaluation report and a controlled baseline benchmark

Estimated effort: 1–2 days plus model runs; depends on M0–M3.

Files: `evaluation/metrics.py`, `evaluation/report.py` (new), `evaluation/plots.py`, `models/baselines.py` (new), `training/benchmark.py` (new).

Reporting:

- Compute RMSE, MAE, signed bias, 95th-percentile absolute error, R², target SD, sample count, trial count, and coverage per subject/trial/position.
- Use equal-position mean RMSE within a subject, then equal-subject averaging for the main aggregate. Also show every subject and its worst-position result; keep pooled R² secondary.
- Mark undefined/unstable R² for constant or nearly constant targets without hiding the corresponding absolute errors. Show the actual torque variation.
- Preserve negative scores and full error ranges in plots. Keep normalized values and Nm correctly labelled; denormalize every split consistently when needed.
- Show variation across seeds separately from variation across participants. Only bootstrap at defensible group/block levels; with five subjects and few repeats, emphasize individual results and describe uncertainty estimates as exploratory.
- Predictions retain real timestamps and trial boundaries. Do not join different trials into an artificial continuous 90-second trace.

Baseline screen, on fixed splits and identical targets:

| Candidate | Configuration | Purpose |
| --- | --- | --- |
| Position-only | Training position means; explicit interpolation policy for unseen angles | Quantify position-related signal |
| Ridge | Current EMG envelopes, angle, EMG × angle interactions; alpha in {0.1, 1, 10} selected on validation | Test static linear/interaction mapping |
| Small MLP | One 16-unit hidden layer; latest feature sample | Test static nonlinearity |
| Existing LSTM | 2 × 8 units, 20 samples, dropout 0.3 | Reference temporal model |

Run the neural screen over three seeds on each eligible subject. Save validation comparisons before evaluating the frozen shortlist on retest. Evaluation masks and coverage must match for paired model comparisons.

Acceptance: a single command creates a machine-readable report and plots, with no confusion between target modes, subjects, seeds, or available positions.

### M5 — Test cross-subject transfer and limited calibration

Estimated effort: 1–2 days plus training; depends on M4.

Files: `training/cross_subject.py`, `training/fine_tune.py`, `preprocessing/splits.py`.

1. Replace pairwise transfer as the primary benchmark with five outer leave-one-subject-out folds. Keep pairwise analysis optional.
2. For each outer fold, train using the four source subjects. Select hyperparameters using source-only validation; if selecting for unseen-subject transfer, use four inner leave-one-source-subject-out folds. Never select using the outer target's retest metrics.
3. Define a source-only preprocessing protocol and, separately, a target-calibrated protocol. Target MVC torque and target-specific normalization require target measurements and cannot be described as zero-calibration transfer.
4. Active-torque folds use only subjects/positions with adequate passive support. The measured-torque benchmark can include every eligible subject. Report the actual contributing subjects per fold.
5. For a bounded adaptation study, use the winning source configuration, dense-only fine-tuning, and labelled calibration budgets of 0, 30, 60, and 120 seconds from the target's original training session. Allocate across up to two positions selected from that session's manifest, never from retest results. Record actual distinct seconds, not overlapping-window counts.
6. Use blocked calibration validation when feasible. For budgets too short for a defensible split, use a training schedule fixed from source-subject simulations; do not early-stop using target retest.
7. Reuse a fitted source model across target evaluations when the source data and configuration are identical.

Acceptance: each outer target remains absent from source training/model selection; adaptation budgets are auditable; each reported transfer score states exactly what target calibration it used.

### M6 — Run focused accuracy and efficiency experiments

Estimated effort: 1–2 days of engineering plus training; depends on a working M4 benchmark. Can proceed alongside M5.

Use sequential validation experiments rather than a full Cartesian sweep:

1. History: 100, 200, 500 ms (10, 20, 50 samples at 100 Hz).
2. Regularization: dropout 0, 0.1, 0.3 on the selected history length.
3. Loss: ordinary MSE versus existing clipped position weighting, after deriving weights strictly from training groups. Compare on the same unweighted reporting metrics.
4. Sensors: four leave-one-muscle-out ablations only for the shortlisted architecture.
5. Efficiency: training stride 1, 5, 10, then batch size 8, 32, 64. Keep evaluation stride at 1. Reassess convergence and scheduler patience because update counts change.
6. Vectorize passive/MVC lookup. Cache only deterministic, split-safe signal processing keyed by file hash, valid interval, filter settings, and split boundaries. Never reuse fold-fitted scales across folds.

Proposed engineering selection rules, fixed before the sweep:

- Prefer a more complex model only if mean validation RMSE improves at least 5% and worst-position RMSE does not deteriorate by more than 5%, with the direction of improvement consistent across at least two of three screening seeds. These are practical comparison rules, not statistical significance claims.
- Accept a speed optimization if median training time improves at least 25% while mean validation RMSE increases no more than 2% and worst-position RMSE increases no more than 5%. Track actual epochs and optimizer updates.
- Retrain only the reference and winning configuration over five seeds for final reporting. If no configuration clears the rules, retain the simpler reference and report that result.
- Do not repeatedly revise these rules after seeing retest results.

Acceptance: an accuracy/runtime/memory comparison identifies whether any tested change earns adoption. No larger LSTM is required by default.

### M7 — Package inference and complete regression checks

Estimated effort: 1 day. Optional causal support adds approximately 1–2 days plus retraining.

Files: `inference.py` (new), `training/run_artifacts.py`, focused tests under `tests/`.

1. Package weights, preprocessing/calibration, channel schema, valid input ranges, target mode, split/manifest hashes, and metrics together.
2. Load the package in a fresh process and verify predictions against saved examples within a declared numerical tolerance.
3. Validate missing channels, non-finite values, short recordings, incompatible sample rates, empty metric groups, and missing calibration support. Produce explicit errors/status rather than plausible-looking outputs.
4. Benchmark warmed-up single-window inference and full preprocessing throughput on the intended machine; record median and p95 latency.
5. If online operation is required, add causal filters with persistent state, remove whole-record demeaning/future-dependent processing, implement streaming resampling, and retrain using the same causal path. Test that modifying future samples cannot change earlier outputs. Report filter delay separately from computation time; the 100 Hz update rate gives a 10 ms compute budget, not a guarantee of 10 ms signal latency.

Acceptance: a complete saved run can reproduce inference without reconstructing undocumented training settings. Offline and causal performance are reported separately if both are implemented.

## Required regression tests

- Manifest overrides are applied before classification; ambiguous records follow the declared inclusion policy.
- Synthetic FLB records preserve channel identity and sample counts; real metadata overrides are explicit.
- Splits contain no shared raw sample ranges and learned preprocessing is invariant to changes in held-out values.
- Missing passive/MVC calibration cannot silently change target semantics; normalization/denormalization round trips recover Nm.
- Repeated preprocessing does not mutate inputs or multiply detected channels.
- Per-position reporting handles missing positions, constant targets, empty groups, and explicit trial boundaries.
- Model/config/preprocessing packages reject incompatible feature layouts and reproduce reference predictions after reload.

## Delivery order and stopping points

1. First delivery: M0–M3, the existing LSTM rerun with auditable data and isolated splits. This is the highest-value milestone.
2. Second delivery: M4–M5, baseline comparisons and transparent subject-transfer evaluation.
3. Third delivery: M6–M7, adopt only demonstrated improvements and package the final pipeline.

Planning estimate: roughly 10–16 engineering days, plus machine-dependent training time; audit/environment issues may extend this. A useful first delivery is feasible before the optional experiments are completed.

The plan remains executable if labels or passive intervals cannot be recovered: primary coverage becomes narrower, measured-torque results remain separate, and sensitivity reports capture the uncertainty. Existing-data validation can support conclusions about these recordings and participants; it cannot establish performance on unobserved participants or sessions.
