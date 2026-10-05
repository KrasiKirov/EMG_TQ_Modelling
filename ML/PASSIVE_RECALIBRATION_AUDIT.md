# Session passive-recalibration audit

The optional audit compares the production/source passive curve with passive
plateaus recorded in a selected session. It does not change the model, target,
or prediction. It uses no active trials and no active torque values.

## Run it

```sh
python -m ML.evaluation.passive_recalibration \
  --subject YES \
  --data-dir /path/to/ML/data \
  --overrides /path/to/reviewed_overrides.json \
  --session retest \
  --output /tmp/YES_passive_recalibration.json \
  --markdown /tmp/YES_passive_recalibration.md
```

The default drift flag is `0.5 Nm`. This is an informational review threshold,
not a correction rule. The JSON records the source and session curves, signed
drift, support status, and the passive evidence used for each point.

## Validation on existing recordings

The audit was run on the available retest passive recordings for HM, EG, and
YES. The most-negative position is p6 under the project position IDs.

| Subject | Retest passive points | RMS drift (Nm) | Maximum drift (Nm) | Flagged points | p6 drift (Nm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| HM | 8 | 0.631 | 0.912 | 5 / 8 | -0.533 |
| EG | 8 | 0.785 | 1.823 | 3 / 8 | +0.082 |
| YES | 6 | 2.626 | 4.936 | 6 / 6 | +3.189 |

The YES p6 result independently confirms the earlier diagnosis: its retest
passive baseline is approximately 3.2 Nm above the source calibration near the
most negative endpoint. EG's p6 passive drift is small, so its p6 error needs a
separate EMG/model-bias investigation.

## Target-correction benchmark

The opt-in pipeline mode was evaluated with the same 500 ms LSTM protocol,
three seeds, and the same source training/validation data. Only retest target
construction changed.

| Subject | Source-only retest RMSE (Nm) | Session-specific retest RMSE (Nm) | Change |
| --- | ---: | ---: | ---: |
| HM | 0.725 | 0.804 | +10.9% |
| EG | 1.699 | 1.545 | -9.1% |
| YES | 2.465 | 1.366 | -44.6% |

At p6, YES RMSE fell from 2.507 to 0.747 Nm and bias changed from -2.483 to
+0.660 Nm. The p6 R² remained unstable because the corrected target still has
very little variation; it changed from -38.640 to -2.584. HM became worse after
the correction, showing that passive recalibration exposes a real session/model
mismatch for that subject rather than universally improving the model.

The option is therefore intentionally not the default. It should be used when
the deployment protocol includes same-session passive calibration, followed by
subject-level validation of both target stability and model bias.

To enable the target correction in the audited benchmark, add:

```sh
python -m ML.training.benchmark ... --target-mode active_torque \
  --session-passive-calibration --evaluate-retest
```

The flag leaves source training and validation targets unchanged and applies the
session-specific passive curve only to retest targets.

## Interpretation and safety

The audit is deliberately diagnostic. A passive difference identifies a changed
target reference; it does not, by itself, identify an additive model-output
correction. Do not subtract the reported drift from predictions automatically.

The production pipeline remains source-session-only. This preserves the current
no-retest-leakage behavior. A future deployment calibration policy may use a
passive retest/session block, but it must be evaluated as a separate protocol and
must not use active retest torque to estimate its own correction.
