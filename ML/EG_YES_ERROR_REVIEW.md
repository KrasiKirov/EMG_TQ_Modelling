# EG/YES positive-angle error review

Date: 2026-10-04

This review uses only the five existing recordings. No new data or labels were
added. The reference is the final within-subject run using active torque,
20-sample history (200 ms at 100 Hz), 60 maximum epochs, patience 12, batch 64,
train stride 5, and seeds 11, 23, 37, 53, and 71. The targeted ablations use
the same preprocessing, reviewed calibration overrides, and training budget,
but use the paired seeds 11, 23, and 37 for a controlled comparison.

## Findings by hypothesis

| Hypothesis | Evidence | Decision |
| --- | --- | --- |
| Incorrect positive/negative labels | The original MATLAB sources consistently declare `maxPF < 0` and `maxDF > 0`. Reinterpreting the labels would not change any numeric prediction or error. | Not supported as the cause. Do not invert the target or position sign. Physical sensor polarity is still a separate verification task. |
| Insufficient positive-angle window count | EG has 8,791 test windows at every position. YES has 8,791 at each evaluated position. | Not a sample-count problem. Positive angular diversity is still limited to two evaluated positive positions, so it remains a coverage limitation rather than a window-count limitation. |
| Subject-specific EMG scaling | Normalization is fitted on training blocks only. EG has 6.4% of test gastrocnemius feature values above its training maximum; YES has 6.7% of test tibialis-anterior values above its maximum. Clipping normalized features to `[0, 1]` improved one representative model by only 0.8% (EG) and 1.2% (YES) in overall RMSE. | A real distribution-shift warning, but not the primary explanation for the positive-angle errors. Add monitoring; do not make clipping the default yet. |
| Low torque variation | EG p2/p8 and YES p7/p8 have high target SD. YES p6 is genuinely low-variance (`SD ≈ 0.40 Nm`) and has a misleadingly negative R². | Low variance explains YES p6, not the main positive-angle failures. Use RMSE/MAE alongside R². |
| Model bias/capacity | EG has systematic negative bias at p2 and positive bias at p8. YES has large negative bias at p7 and p8. The effect persists across seeds. | Supported. The most useful low-risk model-side intervention tested was longer temporal context. |

## Matched retest ablations

The metric is the equal-position mean RMSE within each subject. Lower is better.

| Subject | 20-step baseline | Position weighting | 50-step history | 50-step change |
| --- | ---: | ---: | ---: | ---: |
| EG | 1.961 Nm | 1.873 Nm (-4.5%) | **1.699 Nm (-13.4%)** | consistent improvement in all 3 paired seeds |
| YES | 2.682 Nm | 2.576 Nm (-3.9%) | **2.465 Nm (-8.1%)** | consistent improvement in all 3 paired seeds |

Positive-angle position results:

| Subject/position | 20-step baseline | 50-step history | Change |
| --- | ---: | ---: | ---: |
| EG/p2 | 2.552 Nm | 2.375 Nm | -6.9% |
| EG/p8 | 2.173 Nm | 1.635 Nm | -24.8% |
| YES/p7 | 3.200 Nm | 2.798 Nm | -12.6% |
| YES/p8 | 3.036 Nm | 2.566 Nm | -15.5% |

Position weighting was not selected: it improved some negative-angle positions,
but did not reliably improve the positive-angle positions and was weaker than
the longer-history candidate.

## Selected corrective change

The safest evidence-based change is **not to change the sign convention**. The
50-step history was subsequently validated across all five subjects and is now
the audited pipeline default. The legacy 20-step path remains available as the
historical reference.

The 50-step model increases the measured eager single-window inference time from
roughly 25–35 ms to roughly 58–62 ms. The full five-subject run improved the
equal-subject retest RMSE by 11.3%; HM's worst-position increase was 3.8%, within
the existing 5% acceptance threshold. End-to-end preprocessing and packaged
graph timings still need to be checked on the intended deployment hardware.

The validation-only command used for the selected candidate was:

```sh
python -m ML.training.benchmark \
  --protocol within --subjects EG YES --models lstm \
  --seeds 11 23 37 --epochs 60 --patience 12 --window 50 \
  --data-dir /path/to/ML/data \
  --overrides /path/to/reviewed_overrides.json \
  --evaluate-retest --output ML/runs/eg-yes-window50
```

The retest artifacts remain local and ignored. Full five-subject results are
recorded in `ML/WINDOW50_VALIDATION.md`.
