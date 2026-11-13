# Archived ML code

Code that produced earlier results but is no longer part of either pipeline. Kept so those results can be traced and
re-run. Nothing in `ML/research`, `EXPERIMENTS/pipeline` or `main`'s audited pipeline imports it.

## `experiments/` — the first experiment harness (September–October 2026)

Plugin-based harness that screened model families, losses and gain/drift corrections
(`python -m ML.archive.experiments.run`, queued with `ML.archive.experiments.queue` from the job lists in `jobs/`).
It was replaced by `ML/research` + `EXPERIMENTS/pipeline/run.py`.

Its saved predictions in `experiments/results/` are still read by:

| Reader | What it uses |
| --- | --- |
| `EXPERIMENTS/analysis/rescore_round2.py` | `results/*__F_*` network-type runs, re-scored with the presentation conventions |
| `EXPERIMENTS/dashboard_react/scripts/legacy/build_catalog.py` | `results/*__<model>__fast.npz` for the Exploration entries of the dashboard |
| `EXPERIMENTS/code/verify_handoff.py` | checks the files are present and the right size |

Reports that cite these results: `notes/report/`, `notes/report2/`, `EXPERIMENTS/reports/FINAL_REPORT.md` (network types).
