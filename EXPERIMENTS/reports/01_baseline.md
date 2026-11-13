# Phase 1: baseline reproduction (X00 = frozen original code, learned S1 → tested S2)

Original settings (Nadam 0.003, batch 8, ≤ 200 epochs, ReduceLROnPlateau 0.7/10, EarlyStopping 25). Seeds 0, 1, 2. 18 runs, each 39–145 min on 1 CPU thread.

## Reproduction check against earlier runs of the original code

- **YES −0.49 rad, seed 0:** R² −0.68 (own target). An earlier run of the original code with the original settings, seed 0 (report 1, §7.3), gave −0.68.
- **JM −0.64 rad, seed 0:** R² −0.32, also identical to report 1.
- So the frozen code reproduces last semester's failure.
- **Run-to-run variability is large at the problematic positions.** YES −0.49: −0.68 / −0.26 / −1.80 over the 3 seeds. JM −0.08: −8.47 in one seed (the passive-table bug).
- **The framework copy (X00f) builds identical data but is not bit-identical in training.** The targets differ by rounding at the 1e-6 Nm level (float64 vs float32 arithmetic), and training amplifies that: JM seed 0 stops at epoch 50 instead of 78.
- Results therefore differ at the level of run-to-run variation (JM −0.79: 0.66 in X00 vs −1.92 in X00f, both seed 0).
- **Consequence:** every experiment is compared with **X00f** (same code path, same seeds). X00 remains the official reproduction of last semester.

## Every subject, position and seed (primary key = passive trials of the tested session; original key in brackets)

| subject | position (rad) | R² seed 0 / 1 / 2 | mean R² | MSE (Nm²) | RMSE (Nm) | NRMSE | mean R² on original key |
|---|---|---|---|---|---|---|---|
| CH | -0.737 | 0.96 / 0.91 / 0.98 | 0.95 | 0.998 | 0.955 | 0.086 | 0.95 |
| CH | -0.595 | 0.99 / 0.96 / 0.98 | 0.98 | 1.597 | 1.222 | 0.058 | 0.98 |
| CH | -0.453 | 0.98 / 0.98 / 0.99 | 0.98 | 1.293 | 1.129 | 0.051 | 0.98 |
| CH | -0.326 | 0.92 / 0.90 / 0.91 | 0.91 | 15.616 | 3.945 | 0.114 | 0.91 |
| CH | -0.183 | 0.99 / 0.97 / 0.98 | 0.98 | 4.532 | 2.107 | 0.054 | 0.98 |
| CH | -0.045 | 0.99 / 0.99 / 0.99 | 0.99 | 6.767 | 2.595 | 0.043 | 0.99 |
| CH | +0.097 | 0.98 / 0.98 / 0.98 | 0.98 | 12.504 | 3.536 | 0.052 | 0.98 |
| CH | +0.238 | 0.99 / 0.99 / 0.99 | 0.99 | 7.704 | 2.772 | 0.041 | 0.99 |
| EG | -0.739 | -0.20 / -0.51 / 0.11 | -0.20 | 5.097 | 2.245 | 0.382 | -0.20 |
| EG | -0.598 | 0.45 / 0.78 / 0.89 | 0.71 | 4.067 | 1.912 | 0.209 | 0.71 |
| EG | -0.458 | 0.76 / 0.80 / 0.84 | 0.80 | 4.137 | 2.027 | 0.160 | 0.80 |
| EG | -0.318 | 0.97 / 0.97 / 0.97 | 0.97 | 1.321 | 1.148 | 0.052 | 0.97 |
| EG | -0.177 | 0.96 / 0.96 / 0.97 | 0.96 | 2.608 | 1.614 | 0.071 | 0.96 |
| EG | -0.038 | 0.98 / 0.98 / 0.98 | 0.98 | 2.687 | 1.639 | 0.057 | 0.98 |
| EG | +0.102 | 0.99 / 0.99 / 0.98 | 0.98 | 2.639 | 1.618 | 0.048 | 0.98 |
| EG | +0.247 | 0.98 / 0.97 / 0.97 | 0.97 | 4.189 | 2.042 | 0.064 | 0.97 |
| HM | -0.789 | 0.87 / 0.74 / 0.71 | 0.77 | 0.135 | 0.363 | 0.064 | 0.77 |
| HM | -0.648 | 0.88 / 0.89 / 0.89 | 0.89 | 0.389 | 0.624 | 0.119 | 0.89 |
| HM | -0.505 | 0.97 / 0.97 / 0.97 | 0.97 | 0.106 | 0.326 | 0.060 | 0.97 |
| HM | -0.364 | 0.91 / 0.92 / 0.89 | 0.91 | 1.059 | 1.027 | 0.119 | 0.91 |
| HM | -0.224 | 0.98 / 0.98 / 0.98 | 0.98 | 0.350 | 0.591 | 0.050 | 0.98 |
| HM | -0.084 | 0.98 / 0.95 / 0.97 | 0.97 | 0.982 | 0.980 | 0.063 | 0.97 |
| HM | +0.052 | 0.99 / 0.98 / 0.99 | 0.98 | 0.428 | 0.653 | 0.049 | 0.98 |
| HM | +0.195 | 0.86 / 0.90 / 0.89 | 0.88 | 3.285 | 1.809 | 0.113 | 0.88 |
| IES | -0.591 | 0.16 / 0.60 / 0.52 | 0.43 | 4.530 | 2.100 | 0.254 | 0.43 |
| IES | -0.484 | 0.84 / 0.85 / 0.87 | 0.86 | 1.985 | 1.407 | 0.137 | 0.86 |
| IES | -0.380 | 0.96 / 0.94 / 0.95 | 0.95 | 1.313 | 1.141 | 0.078 | 0.95 |
| IES | -0.274 | 0.95 / 0.97 / 0.96 | 0.96 | 0.695 | 0.827 | 0.064 | 0.96 |
| IES | -0.170 | 0.98 / 0.98 / 0.98 | 0.98 | 1.055 | 1.026 | 0.056 | 0.98 |
| IES | -0.062 | 0.91 / 0.91 / 0.95 | 0.92 | 3.096 | 1.745 | 0.108 | 0.92 |
| IES | +0.039 | 0.96 / 0.94 / 0.95 | 0.95 | 4.431 | 2.096 | 0.085 | 0.95 |
| IES | +0.144 | 0.98 / 0.99 / 0.99 | 0.99 | 1.720 | 1.309 | 0.049 | 0.99 |
| JM | -0.787 | 0.66 / 0.80 / 0.62 | 0.69 | 0.111 | 0.330 | 0.139 | 0.69 |
| JM | -0.644 | -0.32 / 0.42 / 0.16 | 0.09 | 0.557 | 0.736 | 0.286 | 0.09 |
| JM | -0.504 | 0.90 / 0.92 / 0.92 | 0.91 | 0.297 | 0.544 | 0.095 | 0.91 |
| JM | -0.360 | 0.46 / 0.67 / 0.34 | 0.49 | 0.892 | 0.935 | 0.227 | 0.49 |
| JM | -0.218 | 0.90 / 0.93 / 0.91 | 0.91 | 0.155 | 0.394 | 0.086 | 0.91 |
| JM | -0.078 | -55.60 / -44.60 / -52.58 | -50.93 | 145.715 | 12.059 | 2.234 | -2.45 |
| JM | +0.055 | 0.51 / 0.53 / 0.67 | 0.57 | 2.886 | 1.692 | 0.242 | 0.54 |
| JM | +0.199 | 0.69 / 0.72 / 0.79 | 0.74 | 2.312 | 1.516 | 0.183 | 0.74 |
| YES | -0.486 | -0.68 / -0.26 / -1.80 | -0.91 | 0.301 | 0.541 | 0.266 | -0.91 |
| YES | -0.381 | 0.75 / 0.74 / 0.78 | 0.76 | 1.225 | 1.106 | 0.144 | 0.76 |
| YES | -0.278 | 0.95 / 0.92 / 0.96 | 0.94 | 1.248 | 1.106 | 0.087 | 0.94 |
| YES | -0.176 | 0.96 / 0.98 / 0.99 | 0.98 | 1.294 | 1.112 | 0.055 | 0.98 |
| YES | -0.071 | 0.98 / 0.99 / 0.99 | 0.99 | 0.893 | 0.937 | 0.043 | 0.99 |
| YES | +0.031 | 0.99 / 0.99 / 0.99 | 0.99 | 2.036 | 1.427 | 0.041 | 0.99 |
| YES | +0.138 | 0.97 / 0.98 / 0.98 | 0.98 | 5.290 | 2.299 | 0.060 | 0.98 |
| YES | +0.245 | 0.98 / 0.97 / 0.97 | 0.97 | 4.255 | 2.054 | 0.052 | 0.97 |

IES has no passive trials: its key is the raw measured torque (nothing subtracted), the same as the original target.

## What the diagnostics show (details: `02_diagnostics_tables.md`, plots: `plots/X00_*`)

1. **The predictions at the problematic positions are not random.**
   - They follow the timing of the pushes: correlation with the key is 0.80 (YES −0.49), 0.97 (EG −0.74) and 0.86–0.90 (JM).
   - The best alignment lag is about 0 samples, so input and target are aligned correctly.
2. **The error there is mostly OFFSET (the right shape at the wrong level).** Offset is 72 % of the MSE at YES −0.49, 64 % at EG −0.74, 54–71 % at JM −0.64, IES and HM. Wrong push size (scale) is the second part.
3. **The target barely moves there:** the std is 0.40 Nm (YES −0.49), 0.60–0.78 Nm (JM), 2.06 Nm (EG −0.74). A small offset in Nm therefore gives a strongly negative R².
4. **Reference predictor:** the training-session mean at each position gives R² ≈ 0 everywhere. The LSTM is better than that at all positions except YES −0.49, EG −0.74 and JM −0.08 (the table bug).
5. **Training behaviour:** EarlyStopping ends runs after 32–115 epochs (best epoch 7–90). There is no sign of training failure, but the best epoch varies a lot between seeds, which matches the large run-to-run variability.
6. **The 2 worst subjects (rule in `code/diagnostics.py`): YES (mean problematic R² −0.08) and EG (0.25).** They are used for Stage A. X01 (clean passive table) cannot change anything for YES or EG, because their tables have no bad entry, so it is tested on JM.
