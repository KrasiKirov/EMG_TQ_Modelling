# Extreme-position R² review

Date: 2026-10-04  
Scope: 500 ms audited pipeline, within-subject LSTM validation, existing data only  
Subjects: HM, EG, YES

## Executive conclusion

The weak R² at the most negative position has two different causes:

1. **The active-torque target has very little within-position variation.** R² divides
   the squared error by the target variance at that position. A modest absolute
   error can therefore produce a low or negative R². This is the main explanation
   for HM and part of the explanation for the other subjects.
2. **The passive-torque calibration is not stable between sessions.** This changes
   the active-torque label itself when the model is evaluated on retest data. The
   effect is especially large for YES at the most negative position.

This is not evidence that the negative angles are mislabeled as dorsiflexion. In
the project-intended software convention, negative angles are plantarflexion and
positive angles are dorsiflexion. Physical sensor polarity remains unverified.

## Held-out p6 results

The table uses the most negative operating position (`p6`) in the 50-step test
predictions. Values are the mean over seeds 11, 23, and 37. `RMSE / target SD`
shows the error relative to the amount of variation available to explain.

| Subject | Target SD (Nm) | RMSE (Nm) | Bias (Nm) | R² | RMSE / SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| HM | 0.773 | 0.341 | +0.086 | 0.796 | 0.44 |
| EG | 2.062 | 1.691 | -1.433 | 0.311 | 0.82 |
| YES | 0.398 | 2.507 | -2.483 | -38.640 | 6.30 |

The HM result is weak only relative to the high R² at other positions; its error
is small in torque units. YES is a real failure under the current evaluation:
the model error is over six times the target standard deviation, and it is mostly
a systematic offset rather than random variation.

Removing only the held-out mean error as a diagnostic—not as a valid deployment
fix—changes p6 R² to approximately 0.81 for HM, 0.80 for EG, and 0.22 for YES.
This shows that EG is primarily biased, while YES also has a session/label
stability problem and limited predictable variation.

## Passive-calibration evidence

The active target is constructed as:

```text
active torque = measured torque - interpolated passive torque
```

The source-session passive calibration is then applied to the retest session.
At p6, the existing passive recordings show:

| Subject | Source passive calibration at p6 (Nm) | Retest passive plateau (Nm) | Difference (Nm) |
| --- | ---: | ---: | ---: |
| HM | approximately 3.55 | approximately 3.02 | -0.53 |
| EG | approximately 2.54 | approximately 2.62 | +0.08 |
| YES | approximately 7.4 | approximately 10.6 | +3.2 |

For YES, the active target mean at p6 changes from approximately -1.69 Nm in
the training portion to +0.74 Nm in retest when the source calibration is used.
That +2.4 Nm shift is almost exactly the model's held-out bias. The retest passive
plateau indicates that the underlying passive baseline itself moved by roughly
3.2 Nm near the calibration endpoint.

The most negative position is also the endpoint of the passive calibration for
all three subjects. Endpoint interpolation is therefore especially sensitive to
small angle or torque shifts. YES's passive source values are documented
measurements without within-plateau variance, which makes that endpoint less
auditable than the other subjects.

## Measured-torque sensitivity check

Retraining the same model on measured torque instead of passive-subtracted torque
did not solve the p6 problem:

| Subject | Measured-target p6 RMSE (Nm) | Measured-target p6 R² |
| --- | ---: | ---: |
| HM | 0.463 | 0.634 |
| EG | 1.702 | 0.259 |
| YES | 2.773 | -48.332 |

Therefore, changing the target to measured torque is not a safe fix. It is useful
as a sensitivity experiment, but it does not remove the session-specific offset
or the low-variance denominator.

## What can be fixed with the current data

### Safe changes now

- Report per-position RMSE, MAE, bias, target SD, and sample count beside R².
- Mark R² as variance-limited when the target SD is small relative to the error;
  do not use pooled R² to claim that the extreme position is solved.
- Keep negative/positive numeric groups authoritative and label negative values
  as project-intended plantarflexion, not dorsiflexion.
- Retain the 500 ms history: it improves average held-out RMSE, but it does not
  correct passive-calibration drift.
- Add a session-calibration check before interpreting retest performance. The
  current data already contains passive retest trials for this audit.

### A feasible model/pipeline improvement

Add an optional **session passive recalibration** stage. Before interpreting
active retest predictions, estimate the passive torque curve from the available
passive calibration block and recompute a session-specific active-target audit.
This separates a changed passive baseline from a changed active response.

Do not automatically apply the passive difference to the model output: passive
recalibration corrects the target reference, but does not by itself identify a
model-output offset. Any output correction requires an independent calibration
rule and must be tested separately.

This must be evaluated with a leave-one-session-out protocol. It must not use the
active retest torque to estimate its own correction. The current YES p6 result is
strong evidence that this is the highest-value corrective experiment.

### What cannot be reliably fixed without calibration information

No architecture, longer input window, or loss reweighting can recover a
session-specific torque offset that is absent from the EMG features and absent
from the training calibration. For YES p6, an uncalibrated model cannot know
whether the observed torque difference is active muscle torque or a changed
passive baseline.

If a passive/session calibration block is unavailable at deployment, the honest
output is an uncertainty/quality flag for the extreme position—not a silently
corrected torque value.

## Recommended acceptance criteria

For each subject and position, require:

- RMSE and MAE in Nm;
- signed bias in Nm;
- target SD and `RMSE / target SD`;
- R² only as a secondary metric when target variation is sufficient;
- a session-calibration drift check at the most negative position.

The p6 issue should be considered fixed only if a session-calibrated evaluation
reduces the YES/EG p6 bias without degrading the other positions or using
retest active torque labels.
