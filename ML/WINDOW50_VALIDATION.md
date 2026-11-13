# Audited 500 ms history validation

Date: 2026-10-04

The audited pipeline now uses 50 post-downsampled samples as its default history:
50 samples at 100 Hz equals 500 ms. The comparison below uses the same active-
torque target, reviewed calibration overrides, 60-epoch budget, patience 12,
batch 64, train stride 5, and paired seeds 11, 23, and 37 for both configurations.
Only the history length changes.

## Retest results

The primary metric is equal-position mean RMSE within each subject. Lower is
better. Worst-position RMSE is shown as a safety check.

| Subject | 20-step RMSE | 50-step RMSE | Change | 20-step worst | 50-step worst | Worst change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HM | 0.774 | 0.725 | -6.3% | 1.468 | 1.524 | +3.8% |
| EG | 1.961 | 1.699 | -13.4% | 2.552 | 2.375 | -6.9% |
| IES | 1.492 | 1.339 | -10.3% | 2.490 | 2.172 | -12.8% |
| JM | 0.785 | 0.593 | -24.5% | 1.373 | 1.078 | -21.5% |
| YES | 2.682 | 2.465 | -8.1% | 3.218 | 2.798 | -13.0% |
| Equal-subject mean | 1.539 | 1.364 | -11.3% | — | — | — |

All five subjects improved in mean RMSE. HM is the only subject with a higher
worst-position RMSE, and the 3.8% increase remains below the existing 5%
acceptance threshold.

## Runtime and data implications

The input feature format remains five features per timestep:

```text
[gm envelope, gl envelope, soleus envelope, tibialis-anterior envelope, position]
```

Only the shape changes from `(20, 5)` to `(50, 5)`. The existing 90-second
sessions retain ample windows; the longer history removes only the initial
500 ms warm-up region. The LSTM parameter count does not materially increase,
but the observed eager single-window model execution increased from roughly
25–33 ms to 60–62 ms. These timings exclude preprocessing and should be
rechecked with the packaged graph and the intended deployment hardware.

## Decision

The audited pipeline and benchmark CLI now default to 50 steps / 500 ms. This
is an evidence-backed default change, not a sign-convention change. The legacy
dataset builder remains at 20 steps so historical scripts and old comparisons
remain reproducible; it is not the audited evaluation path.

Reproduction command:

```sh
python -m ML.training.benchmark \
  --protocol within --subjects HM EG IES JM YES --models lstm \
  --seeds 11 23 37 --epochs 60 --patience 12 \
  --data-dir /path/to/ML/data \
  --overrides /path/to/reviewed_overrides.json \
  --evaluate-retest --output ML/runs/within-window50
```
