# Stage B summary (2026-10-09): YES + EG, 3 seeds, both directions, reference X00f

> **CORRECTION (2026-10-09, after an independent fact-check):** the numbers below were computed with mismatched seeds. Where the baseline had more seeds than the experiment, all of them were averaged. With MATCHED seeds the Stage A baseline is 0.344 (not 0.22). Then X05 = +0.18 (fails only the noise rule), X04 = +0.09 (fails rule 1) and X03 = −0.01, so **strictly no Stage A experiment qualified for Stage B**. Corrected tables: `stageA_<exp>.md` (regenerated with matched seeds) and `FINAL_REPORT.md`. The YES −0.49 "R² ceiling −2.3" wording is also withdrawn: it is the R² of one hypothetical predictor, not an upper bound (see `results/offset_shift.txt`).


Primary direction = learned S1 → tested S2. Primary key = passive trials of the tested session. Full per-position tables: `stageB_<experiment>.md`.

| experiment | problematic R² (ref 0.156) | cells improved | problematic RMSE change | worst other-position RMSE change | reverse direction Δ | criteria 1/2/3/4 |
|---|---|---|---|---|---|---|
| X05 auto-gain | 0.297 | 3/4 | −0.63 Nm | +9 % | −0.06 | ✔ ✔ ✔ ✘ |
| X04 equal weight per position | 0.420 | 3/4 | −0.67 Nm | +36 % | +0.02 | ✔ ✔ ✘ ✘ |
| X03 learning rate 0.001 | 0.425 | 3/4 | −0.19 Nm | +15.004 % | +0.10 | ✔ ✔ ✘ (by 0.004 %) ✘ |
| C01 auto-gain + equal weight | 0.407 | 3/4 | **−0.77 Nm** | **+4 %** | −0.05 | ✔ ✔ ✔ ✘ |

## Per position, R² per seed (learned S1 → tested S2)

| position | baseline X00f | X05 | X04 | X03 | C01 |
|---|---|---|---|---|---|
| EG −0.74 | 0.01 / −0.18 / −0.79 | 0.92 / 0.91 / 0.89 | 0.94 / 0.95 / 0.96 | 0.43 / −0.13 / 0.22 | **0.96 / 0.96 / 0.97** |
| EG −0.60 | 0.84 / 0.89 / 0.79 | 0.95 / 0.95 / 0.94 | 0.90 / 0.92 / 0.95 | 0.91 / 0.72 / 0.84 | **0.97 / 0.97 / 0.96** |
| YES −0.38 | 0.73 / 0.86 / 0.49 | 0.91 / 0.86 / 0.67 | 0.90 / 0.84 / 0.88 | 0.82 / 0.72 / 0.90 | **0.86 / 0.91 / 0.89** |
| YES −0.49 | −0.52 / 0.12 / −1.36 | 0.16 / −1.44 / −3.16 | −1.00 / −1.01 / −1.20 | −0.43 / −0.35 / 0.47 | −1.36 / −1.20 / −1.00 |

In the reverse direction (learned S2 → tested S1), C01 gives EG −0.74 0.93–0.95 (baseline −0.07 / 0.79 / 0.53), EG −0.60 0.98–0.99, YES −0.38 0.90–0.96, and YES −0.49 −0.69 / −0.29 / −0.39 (baseline 0.40 / 0.57 / 0.15).

## Interpretation (strictly by the pre-written rules)
- **No experiment is confirmed.** All of them fail criterion 4: the mean improvement must exceed 2 × the baseline's run-to-run SD at the problematic positions. Auto-gain and C01 also show the wrong sign in the reverse direction.
- **The reason is one position: YES −0.49.**
  - The offset-ceiling analysis shows its R² cannot exceed about −2.3 under this answer key (learned S1 → tested S2). Its baseline values swing between −1.36 and +0.12 with the seed, which inflates the noise term.
  - Auto-gain makes it worse in both directions. A likely reason: YES barely activates the calf at this angle, so rescaling each muscle by its own range also amplifies noise. Report 1 saw the same effect.
- **At the other 3 problematic positions, C01 is a large, consistent improvement** in every seed and both directions, with no regression elsewhere.
  - EG −0.74 goes from unstable (−0.79 to +0.01) to 0.96–0.97.
  - RMSE at the problematic positions drops by 0.77 Nm.
  - This is reported as an observation, not as a confirmed success under the rules.
- **Decisions:**
  - C01 is the best candidate. It goes to Stage D (the other 4 subjects) as D01, with the JM passive-table fix added.
  - X03 is not combined: it gives smaller RMSE gains, regressions at YES, and noisy results.
  - T01 (passive value as input) is still running and is aimed specifically at YES −0.49.
