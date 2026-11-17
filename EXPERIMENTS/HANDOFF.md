# Handoff: what was transferred and how to check it

Prepared 13 November 2026. For setup, running experiments and updating the dashboard, read the repository
[`README.md`](../README.md) and [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md). This page covers only the artifacts
shipped with the repository and how to verify them.

## What is included

The six FLB recordings, the YES worksheet PDF and the IES metadata DOCX are not in the repository: they are
participant data and the repository is public. Get them privately from the lab and put them in `ML/subjects/`
(ignored by git), or point `EMG_DATA_DIR` at their folder.

| Location | Contents |
| --- | --- |
| `EXPERIMENTS/configs/` | Experiment specs (controlled, normalization, A0 and the calibration variants) |
| `EXPERIMENTS/results/` | Saved predictions and metadata: normalization, A0, the October controlled experiments, and the partial S0/V0 runs |
| `ML/archive/experiments/results/` | Earlier model-family predictions and their metadata |
| `ML/research/results/` | Earlier research predictions, including superseded runs kept under their original labels |
| `EXPERIMENTS/run_collections/window50_retest_map/` | The 60 runs (60-epoch cap) behind the dashboard's `main/session-passive` entry: predictions, models, metrics, histories, manifests, preprocessing, overrides and per-run source snapshots |
| `EXPERIMENTS/run_collections/window50_retest_map_150ep/` | A further 18-run, 150-epoch-cap collection; not on the dashboard |
| `EXPERIMENTS/baseline/retest_map_code/` | An importable copy of the source used by the 60-epoch collection; see its README |
| `EXPERIMENTS/dashboard_react/src/data/` | The dashboard data |
| `EXPERIMENTS/handoff_manifest.json` | Relative path, size and SHA-256 of every transferred artifact |

Earlier prediction archives were recovered from another checkout only when their JSON metadata matched exactly.
Original run metadata, including historical absolute paths, is kept as evidence; those paths are not setup
instructions.

Most runs saved predictions, not trained weights. V0 includes its weights and the run collections include their
`.keras` models. A prediction file alone cannot recover a trained model.

Partial runs: `S0_noshuffle` has 9 completed runs (18 were planned) and V0 has one EG run. Neither is on the dashboard;
do not present them as complete six-person, three-seed studies.

## Check the transfer

From the repository root, with the standard library only:

```bash
python EXPERIMENTS/code/verify_handoff.py           # sizes and SHA-256 checksums
python EXPERIMENTS/code/verify_handoff.py --quick   # paths and sizes only
```

Text files are checked with CRLF normalized to LF, so Windows and Unix checkouts agree; binary files are checked byte
for byte. After deliberately changing artifacts, regenerate the inventory with `--write-manifest`, review its diff,
and commit it with the change. The inventory is a transfer check, not a claim that every planned run exists.

## Checks done at handoff

- All 1,279 prediction archives open with pickled arrays disabled and pass ZIP CRC checks. Saved run ids, subjects,
  seeds and prediction lengths match their metadata.
- The 78 Keras model packages pass ZIP integrity checks; one further HDF5 weight file is checksummed.
- All six recordings load through the research preprocessing with eight active trials per session.
- Re-importing the normalization and 60-epoch retest-map results in an isolated tree reproduced all 1,608
  supplementary dashboard score rows exactly, with identical packed targets and predictions.
- Since then, the spec exporter (`python -m EXPERIMENTS.pipeline.dashboard export`) reproduces every normalization
  score row to within 1e-9 (checked by a regression test, removed in November 2026; see the git history).

## Environment

Training used Python 3.11.1, TensorFlow 2.21.0 and NumPy 2.4.4. Direct dependencies are pinned in
`EXPERIMENTS/requirements-handoff.txt`; it is not a full lock, and numerical results can vary across hardware and
package versions. `EMG_DATA_DIR` defaults to `ML/subjects`; the frozen original-code baseline
(`EXPERIMENTS/code/run_original_baseline.py`) needs it set explicitly.
