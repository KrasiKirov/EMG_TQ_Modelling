# Legacy scoring for the controlled experiments (X, C, D, S, T series)

These scripts built `EXPERIMENTS/results/all_evaluations.csv`, `presentation_runs.csv` and the old single-page dashboard for the
controlled experiments of October 2026 (`EXPERIMENTS/reports/FINAL_REPORT.md`). The current dashboard does not use them:
new experiments go through `python -m EXPERIMENTS.pipeline.dashboard` (see `docs/EXPERIMENTS.md`).

| Script | What it does |
| --- | --- |
| `score.py` | Scores every saved run against the answer key; writes `results/all_evaluations.csv` |
| `presentation_metrics.py` | Per-position median R², percentiles over seeds, MAE and demeaned scores; writes `results/presentation_runs.csv` (used by the deck and the old dashboard) |
| `build_dashboard.py` | Builds the old `EXPERIMENTS/dashboard/` page from the scored results |

Run them from the repository root, e.g. `python EXPERIMENTS/pipeline/legacy/score.py`.
