# EXPERIMENTS: improving LSTM-8 at the plantarflexed positions, one change at a time

**Status (2026-10-09): finished.** Read `reports/FINAL_REPORT.md` first.

**Final candidate D01 = clean passive table + EMG auto-gain + equal weight per position.**
- **All 6 people, original direction, vs the official original-code baseline:** toes-down R² 0.50 → 0.77, RMSE 1.13 → 0.56 Nm.
- **YES at −0.49 rad gets worse.**
- **This candidate was chosen by judgment.** Under the rules written in advance, no change is formally confirmed (`reports/FINAL_REPORT.md` §3).

Other documents:
- `reports/00_audit.md`: the Phase 0 audit;
- `SUCCESS_CRITERIA.md`: the rules fixed before any result;
- `experiment_ledger.csv`: every experiment and its decision.

**Method:** establish the baseline, form a hypothesis, change one thing, measure, interpret, then decide the next step.

## Layout
| path | contents |
|---|---|
| `baseline/original_code/` | frozen copy of the original LSTM-8 code (main checkout working copy), with `original_code_sha256.txt` |
| `configs/` | one JSON per experiment: what it changes relative to LSTM-8, the hypothesis and the motivation |
| `code/run_original_baseline.py` | runs the frozen original code (only adds a seed and saves predictions) |
| `code/run_experiment.py` | runs one experiment (one config × subject × direction × seed) |
| `code/run_queue.py` | runs many jobs in parallel; skips finished ones |
| `code/check_equivalence.py` | shows the experiment framework builds the same data as the original |
| `code/score.py` | scores every run → `results/all_evaluations.csv` |
| `code/diagnostics.py` | Phase 2 diagnostics of the baseline |
| `results/` | per-run predictions (`.npz`), run info (`.json`), and the complete metric tables |
| `plots/`, `reports/`, `logs/`, `literature/` | figures, written interpretations, console logs, verified references |

## Fixed definitions
- **Baseline:** LSTM-8 with the original settings (Nadam 0.003, batch 8, ≤ 200 epochs, ReduceLROnPlateau 0.7/10, EarlyStopping 25), seeds 0, 1 and 2.
- **Directions:**
  - S1 → S2 = learn session 1 ("test"), predict session 2 ("retest"). This is the original protocol and the primary one.
  - S2 → S1 = the reverse, reported as supplementary.
- **Primary answer key:** measured − passive torque from the passive trials of the tested session (clean static holds; YES: worksheet values; IES: none).
- **Metrics per position:**
  - R²;
  - MSE (Nm²);
  - RMSE (Nm);
  - NRMSE = RMSE ÷ (max − min of the key), the original definition.

## Reproduce
```bash
source ML/experiments/env.sh
$PY EXPERIMENTS/code/run_queue.py --original-baseline --subjects YES JM EG IES CH HM --seeds 0 1 2
$PY EXPERIMENTS/code/run_queue.py --exps X01_clean_passive --subjects YES JM --directions S1toS2 S2toS1 --seeds 0 1
$PY EXPERIMENTS/code/score.py
$PY EXPERIMENTS/code/diagnostics.py
```
