# Stage A summary (2026-10-09, subjects YES + EG, learned S1 → tested S2, seeds 0–1, reference X00f)

> **CORRECTION (2026-10-09, after an independent fact-check):** the numbers below were computed with mismatched seeds. Where the baseline had more seeds than the experiment, all of them were averaged. With MATCHED seeds the Stage A baseline is 0.344 (not 0.22). Then X05 = +0.18 (fails only the noise rule), X04 = +0.09 (fails rule 1) and X03 = −0.01, so **strictly no Stage A experiment qualified for Stage B**. Corrected tables: `stageA_<exp>.md` (regenerated with matched seeds) and `FINAL_REPORT.md`. The YES −0.49 "R² ceiling −2.3" wording is also withdrawn: it is the R² of one hypothetical predictor, not an upper bound (see `results/offset_shift.txt`).


| experiment | what changed | problematic R² (ref 0.22) | cells improved | problematic RMSE change | worst other-position RMSE change | criteria 1/2/3/4 | decision |
|---|---|---|---|---|---|---|---|
| X05 auto-gain | EMG rescaled by each muscle's recent range | **0.53** | 3/4 | −0.63 Nm | +4 % | ✔ ✔ ✔ ✘ | Stage B (only fails the noise test with 2 seeds) |
| X04 equal weight per position | each position's target ÷ its std | 0.43 | 3/4 | −0.56 Nm | +37 % (YES +0.03) | ✔ ✔ ✘ ✘ | Stage B |
| X03 learning rate 0.001 | 0.003 → 0.001 | 0.34 | 3/4 | −0.06 Nm | +33 % (YES −0.28) | ✔ ✔ ✘ ✘ | Stage B |
| X02a/b/c dropout 0 / 0.1 / 0.5 | 0.3 → 0 / 0.1 / 0.5 | 0.17 / 0.16 / 0.17 | ≤ 2/4 | worse | up to +88 % | ✘ | rejected |
| X06 0.4 s window | 20 → 40 samples | −0.21 | 1/4 | worse | +61 % | ✘ | rejected |
| X01 clean passive table (JM only) | static holds only | JM −0.08: −54.9 → 0.58 | – | 12.5 → 1.08 Nm | – | – | kept as a bug fix |

## What we learned
- **The baseline's toes-down failure is not overfitting or a learning-rate problem.** Changing dropout in either direction makes things worse; a smaller learning rate helps a little but hurts other positions.
- **The two changes that address diagnosed causes help most at the toes-down positions:**
  - auto-gain, for the EMG size change between sessions;
  - equal weight per position, for the loss imbalance.
  - Both fix EG −0.74 (R² −0.09 → 0.92 / 0.95).
- **A longer window does not help with the original settings,** unlike the earlier batch-64 study. Window length is not the bottleneck.
- **YES −0.49 (learned S1 → tested S2) is not helped by any change.** The offset-ceiling analysis (`results/offset_ceiling.txt`) explains why:
  - under the passive-trial key, the resting level of the target shifts by 0.72 Nm between the sessions;
  - the whole signal there only varies by 0.40 Nm (std);
  - so even perfect push tracking cannot exceed R² ≈ −2.3 without information about that shift.
  - This motivates Tier 2 experiment T01: the passive-trial value as an extra input.

## Next
- **Stage B:** 3rd seed + reverse direction for X05, X04, X03.
- **Stage C1:** X05 + X04, started in parallel.
- **T01 (Tier 2):** starts when CPU load allows.
