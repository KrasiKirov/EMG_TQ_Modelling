# Implementation validation — 2026-09-25

## Scope and interpretation

All experiments used the five existing local recordings. No new participants,
recordings, or external training data were added. Raw recordings, worksheets,
checkpoints, and participant-level predictions are not included in this commit.
Full local artifacts are under the ignored `ML/runs/` directory in the isolated
fork worktree, with source snapshots, configurations, manifests, and file hashes.

These are retrospective internal results, not prospective or external validation.
The revised preprocessing and target definitions differ from the historical
pipeline, so the numbers below do not establish an accuracy gain over the old
saved checkpoints. They compare models under the same revised protocol.

## Accuracy and design

The validation-only screen included position-only, ridge, a static 16-unit MLP,
and the existing 2 × 8-unit LSTM. Neural seeds were 11, 23, and 37; training used
at most 60 epochs, patience 12, batch 64, stride 5, and 20-sample history.
All evaluation windows used stride 1. The target was active torque in Nm, using
training-session passive calibration. Existing IES metadata and a locally reviewed
YES worksheet transcription supplied missing calibration measurements.

| Model | Validation macro RMSE (Nm) |
| --- | ---: |
| Position-only | 5.545 |
| Ridge | 2.469 |
| Small MLP, 3-seed mean | 1.969 |
| Reference LSTM, 3-seed mean | 0.784 |

The LSTM was retained without increasing its 1,001 parameters. The final run used
seeds 11, 23, 37, 53, and 71, with the same configuration and explicit retest
evaluation. Mean RMSE weights positions equally within each subject, then subjects
equally. Seed SD is variation of that aggregate, not population uncertainty.

| Retest subject | Ridge RMSE (Nm) | LSTM 5-seed mean RMSE (Nm) | LSTM mean worst-position RMSE (Nm) |
| --- | ---: | ---: | ---: |
| HM | 1.740 | 0.812 | 1.572 |
| EG | 4.056 | 1.936 | 2.573 |
| IES | 2.975 | 1.444 | 2.428 |
| JM | 1.501 | 0.777 | 1.441 |
| YES | 6.674 | 2.631 | 3.214 |
| Equal-subject mean | 3.389 | 1.520 | — |

LSTM validation macro RMSE was 0.780 ± 0.013 Nm across five seeds; retest macro
RMSE was 1.520 ± 0.042 Nm. The validation-to-retest gap remains material. There
are eight evaluated retest positions for four subjects but only four for YES:
records 27, 29, 31, and 33 remain ambiguous and excluded. Missing positions do not
count as zero error. The YES worksheet does not independently resolve their labels.

## Transfer is still the principal weakness

A separate measured-torque pilot trained each source model on four subjects and
held out the fifth. It used a fixed configuration, seed 11, at most 30 epochs,
and source-only validation. No target torque labels or target-fitted EMG scales
were used in the zero-calibration predictions. Adaptation then used only prefixes
from two target training-session positions, with a frozen backbone and final-layer
fine-tuning. Retest was never used for stopping or position selection.

| Held-out subject | Zero-calibration RMSE (Nm) | 120-second adaptation RMSE (Nm) |
| --- | ---: | ---: |
| HM | 2.032 | 1.935 |
| EG | 9.589 | 9.484 |
| IES | 2.257 | 2.474 |
| JM | 1.871 | 1.695 |
| YES | 6.116 | 5.516 |
| Equal-subject mean | 4.373 | 4.221 |

The 30- and 60-second conditions were also executed and retained locally. Brief
adaptation is not reliably beneficial: IES worsened, and EG retained large errors.
This pilot does not support deploying one general model across participants.
Do not compare this table directly with active-torque within-subject scores.

## Efficiency

- The reference model remains small: 1,001 parameters and 20 × 5 inputs.
- Preprocessing took 0.45–0.66 seconds per subject in the final run. Feature/target
  arrays occupied approximately 23.9–38.2 MB per subject, excluding metadata,
  temporary buffers, and framework memory.
- Median LSTM training time was 33.4 seconds per subject/seed. Some experiments
  ran concurrently, so these are observed timings, not controlled speed claims.
- The packaged loader caches a TensorFlow graph. In 200 alternating measurements
  on the same HM checkpoint, warmed single-window median execution fell from
  26.27 ms (eager) to 0.565 ms (graph); graph p95 was 0.684 ms. Initial graph
  compilation/first execution took 80.9 ms. The timing input was synthetic and
  already preprocessed; no training ran during this profile.
- Graph inference preserved all checked saved predictions. This computation
  improvement does not make zero-phase filtering causal or remove filter delay.

The full history/dropout/weighting/sensor/stride/batch research sweep has not been
completed. All six stages ran in a 14-candidate, one-epoch smoke test only; its
tentative selections were not adopted. Defaults remain engineering choices.

## Reliability checks

- 30 unit/regression tests: disjoint raw filter support, held-out invariance,
  prefix-bounded adaptation, manifest overrides, finite-value handling, explicit
  missing calibration, nonmutating preprocessing, binary fixtures, grouping,
  sweep acceptance rules, and baseline/neural serialization.
- Fresh-process verification of all 35 final within-subject packages and all 20
  transfer/adaptation packages: 3,481,236 saved predictions reproduced with zero
  observed difference. The verifier enforces 1e-5 Nm absolute / 1e-6 relative
  tolerance and reconstructs adaptation prefixes before normalization.
- Historical reconstruction loaded all five old checkpoints; each has 1,001
  parameters and a 20 × 5 input. Their training configuration cannot be recovered
  reliably, so reconstructed historical results remain separately labelled.
- The installed locked environment passed `pip check`; Python source parsing,
  entry-point help, and Git whitespace checks passed. TensorFlow/Keras emit
  third-party deprecation warnings, but no test failures.

## Next work, still using existing data

1. Run the full validation-only sequential sweep over all five subjects and at
   least three seeds. Adopt only changes that pass the predeclared accuracy,
   worst-position, and runtime rules; do not select from retest outcomes.
2. Expand transfer to multiple seeds. Any transfer-specific hyperparameter search
   must use inner source-subject folds, not the outer held-out participant.
3. Register sensitivity checks for passive-calibration support and filter-edge
   trimming. Preserve uncertainty in the unresolved YES records; do not infer
   their labels from whichever choice produces the best score.
4. Check real-file parity against the original MATLAB FLB helper if it becomes
   available locally. Binary fixtures are not a substitute for that comparison.
5. Add causal filtering/stateful streaming only if online operation is required,
   with separate retraining and future-sample invariance tests.

Local artifact directories: `ml-validation-screen-20260925`,
`ml-within-final-20260925`, `ml-loso-measured-baselines-20260925`,
`ml-transfer-pilot-20260925`, `ml-reconstructed-reference-20260925`, and
`ml-sweep-smoke-20260925`. See each run's configuration before reproducing it;
the reviewed calibration override is copied into the final run.
