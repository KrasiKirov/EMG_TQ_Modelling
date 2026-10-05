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

## Interpretation and safety

The audit is deliberately diagnostic. A passive difference identifies a changed
target reference; it does not, by itself, identify an additive model-output
correction. Do not subtract the reported drift from predictions automatically.

The production pipeline remains source-session-only. This preserves the current
no-retest-leakage behavior. A future deployment calibration policy may use a
passive retest/session block, but it must be evaluated as a separate protocol and
must not use active retest torque to estimate its own correction.
