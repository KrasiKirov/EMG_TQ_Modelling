# Success criteria (written 2026-10-08, BEFORE any EXPERIMENTS/ result was produced)

These rules decide whether an experiment counts as an improvement. They are fixed now, so they cannot be adjusted afterwards to favour a model.

## What is compared
- **Reference:** the original LSTM-8 baseline (X00), using the same subjects, positions, direction and seeds.
- **Primary direction:** the original protocol. The model learns from session 1 ("test") and is tested on session 2 ("retest"), which it never saw. The reverse direction (S2 → S1) is reported as supplementary evidence.
- **Primary answer key:** passive trials of the tested session, i.e. measured torque − passive torque from the clean static holds of the tested session's passive trials (YES: lab worksheet values).
  - This follows the supervisor's decision that passive torque comes from the passive trials.
  - The original target (the original passive table, which contains JM's bad entry) is always reported next to it.
- **Problematic positions:** the 2 most plantarflexed (most negative) positions of each subject.
- **Other positions:** the remaining 6 positions of each subject.

## Metrics
Reported for every subject, position, direction and seed:
- **R²** = 1 − Σ(pred − key)² / Σ(key − mean)²
- **MSE** in Nm²
- **RMSE** in Nm
- **NRMSE** = RMSE ÷ (max − min of the key at that position). This is the original project's definition (ML/evaluation/metrics.py).

## Stage A (screening, 2 worst subjects): an experiment is "promising" if all of these hold
1. **Problematic positions improve.** The seed-mean R² at the problematic positions, averaged over the 2 subjects, rises by at least **0.10**, and the mean RMSE there does not increase.
2. **The improvement is consistent.** R² improves in at least **3 of the 4** (subject × problematic position) cells, in the primary direction.
3. **No serious regression elsewhere:**
   - no other position loses more than **0.05** seed-mean R²;
   - the mean R² of the other positions drops by no more than **0.01**;
   - no other position's RMSE rises by more than **15 %**.
4. **Larger than run-to-run noise.** The mean problematic-position improvement is larger than **2 ×** the baseline's run-to-run standard deviation there.

If an experiment meets 1–2 but fails 3, it is marked "trade-off" and is not combined without discussion.

## Stage B (confirmation)
Promising experiments are repeated with 3 seeds. They must still meet the Stage A rules, and they must show the same sign in the supplementary direction.

## Stage C (combinations)
Changes are added one at a time to the best configuration. A step is kept only if it meets rules 1–4 relative to the configuration *before* that step, using the same subjects and seeds.

## Stage D (validation on the other 4 subjects)
The configuration selected at the end of Stage C is run on the 4 subjects not used for screening, in both directions, with 3 seeds. Its results are reported whatever they are. They are not used to change the selection. If it fails there, that is reported as a limitation.

## Tie-breaking
Prefer the configuration with **fewer changes from LSTM-8**. Then prefer smaller run-to-run variability.

## Also always reported (never hidden behind averages)
- every per-subject, per-position, per-seed value;
- regressions at any position;
- target variance per position;
- a reference predictor: the training session's mean target at that position (uses no test information).
