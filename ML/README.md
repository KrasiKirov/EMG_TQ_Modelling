# Audited EMG-to-torque pipeline

This pipeline uses the existing five recordings. It preserves raw trial identity,
isolates filtering and normalization across splits, distinguishes measured torque
from estimated active torque, and saves reproducible model/preprocessing packages.
No additional recordings are required. The previous scripts and artifacts remain
available for historical comparison; historical scores are not directly comparable
to the new target/split definitions.

## Environment and commands

Run commands from the repository root (the directory containing `ML`). Python
3.13.2 and the versions in `ML/requirements.lock` were used for verification on
macOS arm64. Use a project virtual environment and install that lock file.
`EMG_DATA_DIR` or `--data-dir` can point to the existing data directory; no copying
of the study recordings into a new checkout is necessary.

```sh
python -m unittest discover -s ML/tests -v
python -m ML.preprocessing.trial_manifest --output ML/runs/audit
python -m ML.training.benchmark --models position ridge mlp lstm --output ML/runs/screen
python -m ML.training.benchmark --protocol loso --models ridge lstm --output ML/runs/loso
python -m ML.training.benchmark --protocol adaptation --models lstm --output ML/runs/adaptation
python -m ML.training.sweep --stages history dropout loss stride batch --output ML/runs/sweep
python -m ML.training.verify_run --run ML/runs/final --output ML/runs/final/reload_verification.json
python -m ML.training.profile_inference --package ML/runs/final/HM_lstm_seed11 --output ML/runs/final/inference_profile.json
```

Outputs must not already exist. The entry points `ML.training.train`,
`ML.training.cross_subject`, and `ML.training.fine_tune` now invoke the audited
runner with within-subject, LOSO, and adaptation defaults respectively. Use
`--help` for their new shared CLI. Older argument combinations are not silently
translated into different protocols. Old plotting/diagnostic and capacity scripts
still use the historical builder; they are not the audited evaluation path.

The default is three neural seeds, 200 maximum epochs, early stopping, batch 64,
and training stride 5; evaluation stride remains 1. These are explicit engineering
defaults, not a claim that the efficiency sweep has selected them. Torque is
standardized using training mean/SD for neural optimization, then returned to Nm
for all reports. `history.json` states the standardized loss units.

## Trial identity and calibration

Every run saves the source-file hash, trial manifest, exclusion reasons, coverage,
and fitted preprocessing. Labels parsed from comments are marked as such; they
are not represented as independently verified. The inventory retains channel
range, RMS, endpoint concentration, and adjacent-step statistics for quality
review; these statistics alone do not establish clipping or sensor failure.
YES records 27, 29, 31, and 33
remain ambiguous and are excluded. Position stability is checked independently
of torque variation: varying torque can still be isometric.

The active-torque target is `measured - fixed passive calibration`. Only the
initial stable plateau in permitted training-session passive intervals is used;
later quiet fragments of corrupted/nonstationary tails are not accepted by default.
The explicit heuristic checks are torque SD <=0.5 Nm, short-bin position range
<=0.01 rad, whole-plateau range <=0.02 rad, and duration >=3 seconds. These are
documented quality-screening choices, not physiological guarantees. Per-channel
EMG RMS is retained for review; a passive comment alone does not prove relaxation.
Unsupported positions produce unavailable active-torque samples, never a zero
passive fallback. Interpolation is linear; endpoint clamping is allowed only
within a recorded 0.015 rad measurement tolerance.

The existing IES DOCX contains a passive-torque table. It is parsed directly from
the local document, checked for eight unique positions, and hashed as evidence.
The YES PDF is a scanned worksheet: a reviewed transcription can be supplied
through `--overrides path/to/local-overrides.json`. The local transcription used
for the implementation experiments is kept with local run artifacts, not published
as study data. All raw data, metadata documents, and run predictions stay local.

Override structure (example values are illustrative, not study measurements):

```json
{
  "SUBJECT": {
    "12": {
      "kind": "active", "session": "test", "position_id": "p2",
      "status": "included", "valid_intervals": [[1000, 80000]],
      "evidence": "Specific existing record establishing the correction"
    },
    "passive_calibration": {
      "positions": [-0.5, 0.2], "torques": [2.0, -3.0],
      "evidence": [{"source": "worksheet.pdf", "sha256": "actual-file-hash",
                    "session": "test", "table": "reviewed source location"}]
    }
  }
}
```

Calibration sources are hash-checked and must identify training/pre-test evidence.
Run `--target-mode measured_torque` for a separate benchmark without passive
subtraction. Do not pool the two target modes into one result.

## Validation and transfer semantics

Raw test-session trials are split into chronological 80/20 blocks with a one-second
boundary exclusion. Blocks are filtered independently, downsampled with line
padding, and trimmed one second at both ends before windowing. `--edge-trim` and
`--gap` allow registered sensitivity checks. The trim is a declared engineering
choice; it is not a proof that every recording is artifact-free. EMG scale is fit
only on processed training blocks. Windows retain complete raw filter-support
ranges, their own sample ranges, and original trial timestamps.

Reports show per-trial/per-position errors, target variation, low-variance flags,
and coverage. The main comparison is equal-position RMSE within each subject,
then equal-subject averaging. Pooled R² is secondary. A missing position is not a
zero-error position. Seed variability and subject variability are distinct.

The LOSO runner fits on source subjects only and selects ridge strength on
source-only validation. Neural hyperparameters are fixed by the invocation;
this is not a nested hyperparameter search. Do not choose configurations from
outer held-out results. For a cross-subject hyperparameter search, run a separate
inner source-subject validation study first. Target EMG normalization is undone
and replaced by source-training scales. Active-torque transfer still requires
declared target passive calibration; measured-torque transfer does not use target
torque labels or target-fitted EMG scales.

Adaptation uses disjoint raw prefixes from up to two target test-session positions,
selected without retest outcomes. Budgets count unique recorded active seconds,
including filtered/discarded portions, and are separate from passive calibration.
Prefixes are rebuilt before filtering so future samples outside the budget cannot
affect calibration. Only the final dense layer is fine-tuned; insufficient budgets
are explicitly reported as unavailable.

Retest predictions are generated only with `--evaluate-retest`. Since these
recordings' results were already inspected, retest reports remain retrospective
internal validation, not evidence of a pristine external test set.

## Experiments and inference

The sequential sweep covers 100/200/500 ms history, dropout 0/0.1/0.3, position
weighting, muscle ablations, stride 1/5/10, and batch 8/32/64. Selection uses
validation only: >=5% mean improvement with <=5% worst-position deterioration;
efficiency choices require >=25% faster training with <=2% mean error increase.
These are engineering thresholds, not statistical significance tests. Sensor
ablations are diagnostic and never automatically remove sensors.

```python
from ML.inference import Predictor
predictor = Predictor("ML/runs/final/HM_lstm_seed11")
# frame has position, gm, gl, sol, ta and attrs['domainIncr']; torque is optional.
result = predictor.predict_recording(frame)
```

Packages include fitted scales, passive support, target definition, model shape,
and source/manifest hashes. Inference is explicitly offline: zero-phase filters
use future samples. Timing reports measure warmed-up inference only, not envelope
delay. A causal streaming implementation requires a separately validated model;
it is not claimed by this release.

The loader caches a TensorFlow inference graph with a variable batch dimension.
`profile_inference` compares warmed-up eager and graph execution on the same
weights; its single-window latency excludes signal processing. `verify_run`
reconstructs saved retest windows (including calibration prefixes for adaptation)
and checks predictions at 1e-5 Nm absolute / 1e-6 relative tolerance in a fresh
process. It never refits the model or modifies the saved checkpoint.
