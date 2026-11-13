# Teammate handoff: experimentation-normalization

Prepared 13 November 2026, preserving the dashboard at `c2273f3`.

Your work is to improve the experiment-to-dashboard pipeline, make experiment
components and navigation derive from data, run further experiments, clean up the
code, and document the resulting workflow. The bundled dashboard is a reference
snapshot to preserve while doing that work. The handoff adds existing artifacts;
it does not retrain models or publish additional experiments to the interface.

## Open the dashboard from a fresh clone

Check out `experimentation-normalization` after cloning. Open
`EXPERIMENTS/dashboard_react/dist/index.html` directly in a browser. This one file
includes the UI, scores and sampled waveforms; it needs no server or Python.

For React development, with Node.js 18 or newer installed:

```powershell
cd EXPERIMENTS/dashboard_react
npm.cmd ci
npm.cmd run dev
```

Use the localhost URL printed by Vite. To rebuild the standalone HTML:

```powershell
npm.cmd run build
```

On macOS/Linux use `npm` in place of `npm.cmd`. The build runs the TypeScript
check before bundling. It does not import new Python experiment results.

## What is included

| Location | Contents |
| --- | --- |
| `ML/subjects/` | Six original FLB recordings, the YES worksheet PDF, and the IES metadata DOCX used for calibration provenance |
| `EXPERIMENTS/configs/` | Controlled and normalization experiment specifications |
| `EXPERIMENTS/results/` | Saved predictions and metadata, including normalization, A0, older controlled experiments, and the completed S0/V0 runs |
| `ML/experiments/results/` | Earlier model-family predictions and their existing metadata |
| `ML/research/results/` | Earlier research predictions, including explicitly superseded runs kept under their original labels |
| `EXPERIMENTS/run_collections/window50_retest_map/` | The 60 completed 60-epoch-cap runs behind `main/session-passive`: predictions, models, metrics, histories, manifests, preprocessing, overrides and per-run source snapshots |
| `EXPERIMENTS/run_collections/window50_retest_map_150ep/` | An additional 18-run, 150-epoch-cap collection, retained separately; it is not currently registered as a dashboard experiment |
| `EXPERIMENTS/baseline/retest_map_code/` | A convenient importable copy of the source from the 60-epoch collection; see its README |
| `EXPERIMENTS/dashboard_react/src/data/` | The existing dashboard datasets |
| `EXPERIMENTS/handoff_manifest.json` | Relative paths, sizes and SHA-256 checksums of the transferred artifacts |

The archive is intentionally substantial. Earlier prediction archives were recovered
from another checkout only when their adjacent JSON metadata matched this branch
exactly. Original run metadata, including historical absolute paths, is retained
as evidence; those paths are not setup instructions for your machine.

Most controlled/normalization runs saved predictions rather than trained weights.
The V0 example includes its saved weights; the benchmark collections include their
`.keras` models. A prediction file alone cannot recover a trained model.

Check the transfer from the repository root (standard-library Python only):

```powershell
python EXPERIMENTS/code/verify_handoff.py
```

Text checksums/sizes normalize CRLF to LF so Windows and Unix checkouts agree;
binary artifacts are checked byte for byte. `--quick` checks paths and sizes only.
After intentional artifact changes, regenerate
the inventory with `--write-manifest`, review its diff, and commit it with the changes.
The inventory is a transfer check, not a declaration that every planned run exists.

## Python setup and a new experiment

The recorded training environment used Python 3.11.1, TensorFlow 2.21.0 and NumPy
2.4.4. Direct scientific dependencies are pinned in `requirements-handoff.txt`;
this is not a full transitive lock. Run JSONs and benchmark provenance retain their
recorded versions. Numerical training results can vary across hardware and packages.

From the repository root on Windows, with Python 3.11 available:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r EXPERIMENTS/requirements-handoff.txt
$env:EMG_DATA_DIR = (Resolve-Path .\ML\subjects).Path
$env:PYTHONPATH = (Get-Location).Path
```

The data-directory environment variable is also needed by the frozen original-code
baseline wrapper. The main framework otherwise defaults to `ML/subjects`.
On macOS/Linux create the environment with `python3.11 -m venv .venv` and use
`.venv/bin/python`; set the same environment variables with your shell's syntax.

Do not source the old `ML/experiments/env.sh` or `normalization/env.sh` unchanged:
they contain the author's worktree and cache paths. Python preprocessing creates
its own local cache; no cache or virtual environment needs to be copied.

For an A0 follow-up, create a new, unused configuration file, for example
`EXPERIMENTS/configs/TEAM01_active_dropout.json`:

```json
{
  "id": "TEAM01_active_dropout",
  "title": "Balanced positions with dropout 0.2",
  "hypothesis": "Test a reduction in dropout while retaining the A0 target and scaling.",
  "changes": {
    "passive": "static",
    "emg": "trainmax",
    "balance": "scale",
    "window": 20,
    "dropout": 0.2
  }
}
```

This is an example, not an experiment already run. `changes` overrides the shared
`BASE` in `EXPERIMENTS/code/run_experiment.py`; there is no inheritance from A0.
Keep the filename and `id` consistent, and never reuse an existing run ID for a
different configuration.

```powershell
& .\.venv\Scripts\python.exe EXPERIMENTS/code/run_experiment.py --exp TEAM01_active_dropout --subject EG --direction S1toS2 --seed 0 --threads 1
```

This is a full training run, not a quick installation test. It writes
`EXPERIMENTS/results/TEAM01_active_dropout/EG/S1toS2/seed0.npz` and `.json`.
The NPZ contains `y_own_target`, `y_pred`, `trial`, `angle`, `session` and `t`.
The JSON records the resolved configuration, seed, window counts and training info.
Use `--save-weights` if trained weights are required; new checkpoints remain ignored
until deliberately selected for version control.

For several runs, the existing queue accepts `--exps`, `--subjects`, `--directions`,
`--seeds` and `--workers`. It skips jobs based on result-JSON existence and does not
check that the requested configuration still matches that JSON. Inspect failures
and coverage explicitly: `QUEUE_DONE` does not mean all jobs succeeded.

## How results currently reach React

```text
recordings + configuration
  -> Python training -> prediction NPZ + metadata JSON
  -> Python scoring/export -> dashboard JSON
  -> React selectors/aggregations -> charts, tables and CSV exports
```

The browser loads `results.json` through `src/dataset.ts` and
`presentation-catalog.json` through `src/lib/catalog.ts`. It never scans the run
folders, trains a model, or reads FLB/NPZ files. `catalog.json` is an intermediate
historical archive, not a direct client import. Exporters calculate run-level
scores; React aggregates them for the selected people, positions and seeds.
Waveforms are sampled/packed for display; score the full saved arrays, not traces
decoded from the dashboard.

There are currently multiple export paths:

| Results | Current path |
| --- | --- |
| Older controlled experiments | `code/presentation_metrics.py` -> `code/build_dashboard.py` -> `results.json` |
| Historical families/research | `dashboard_react/scripts/build_catalog.py` -> `catalog.json` -> `build_presentation_catalog.py` |
| Normalization and A0, including 10 s calibration | `dashboard_react/scripts/import_normalization.py` -> `presentation-catalog.json` |
| The displayed 60-epoch retest-map collection | `dashboard_react/scripts/import_retest_map.py EXPERIMENTS/run_collections/window50_retest_map` -> `presentation-catalog.json` |

All script paths above are relative to `EXPERIMENTS/`. Run the retest importer from
the repository root using its full path. It recomputes metrics and compares them
with the saved `metrics.json`. Its experiment ID and explanatory text are fixed to
the displayed 60-epoch method: do not import the 150-epoch collection through it
as though that were the same experiment.

To refresh the *currently supported normalization entries* without training:

```powershell
& .\.venv\Scripts\python.exe EXPERIMENTS/normalization/calibration_test.py
& .\.venv\Scripts\python.exe EXPERIMENTS/normalization/calibration_test.py --exp A0_active_trainmax
& .\.venv\Scripts\python.exe EXPERIMENTS/dashboard_react/scripts/import_normalization.py
```

Then run `npm.cmd run build` in `EXPERIMENTS/dashboard_react` and review the result.
This importer is a study-specific registry, not a generic new-experiment importer.

Important limits when changing the pipeline:

- `build_presentation_catalog.py` replaces the focused catalogue. It preserves the
  imported retest-map collection specifically, but removes normalization entries
  until they are imported again. Its historical report import also depends on
  documents at `origin/main`, making a full rebuild sensitive to that moving ref.
- The old scoring script discovers only certain directory-name prefixes and
  reconstructs the common clean-passive target. It is not the exporter for arbitrary
  total-torque experiments. The normalization importer uses each saved run's target.
- `src/lib/presentation.ts`, `src/lib/experimentStory.ts`,
  `src/tabs/AllExperimentsTab.tsx` and parts of `ExperimentTab.tsx` contain explicit
  IDs, labels or assumptions. Merely adding a result directory will not populate
  every menu/table. Study-specific explanatory text can become wrong if settings
  change without updating it.
- `S0_noshuffle` has 9 completed runs (not the planned 18), and V0 has one EG run.
  Neither is registered in the current loaded dashboard datasets. Retain their
  partial coverage; do not imply a complete six-person, three-seed study.
- `ML/research/README.md` and older reports describe earlier recommendations,
  including auto-gain. Read `normalization/PLAN.md` and `normalization/REPORT.md`
  for the later study context rather than treating every historical "final" label
  as the current model choice.

## Suggested order of work

1. Verify the artifacts and build/open the existing dashboard before refactoring.
2. Define one versioned experiment/run schema. Include the resolved config,
   subject/direction/seed, source revision and input hashes, target definition,
   calibration protocol, scoring interval and actual coverage.
3. Separate model-specific run adapters from shared scoring and export code. Make
   validation fail clearly on missing/misaligned arrays, duplicate run IDs and
   inconsistent configs. Partial studies should remain explicitly partial.
4. Derive React groups, selectors, experiment summaries and configuration displays
   from exported metadata. New experiments supported by the common schema should
   not need React edits. Keep scientific interpretation as authored metadata.
5. Add one documented run/validate/export/build workflow. Use a small known run as
   a regression fixture, preserving existing metrics and target definitions while
   changing the plumbing.
6. Run additional experiments under new IDs, then remove obsolete code only after
   tracing its callers and preserving the provenance required by the reports.

Keep active- and total-torque targets explicit. Calibrated entries use measured
test-session torque for calibration and are scored from 30 s onward; uncalibrated
entries generally use the full available trial. Those are different protocols.
Avoid interpreting a cross-protocol comparison as an isolated model improvement.

For each intentional dashboard update, commit the new specification and run
artifacts, generated JSON, rebuilt HTML, and any documentation/manifest updates
together. Check the result from an exported tree or fresh clone, not only from an
author's worktree with extra local files.

## Checks at handoff

All 1,279 prediction archives were opened with pickled arrays disabled and their
ZIP CRCs checked. Saved controlled-run IDs, subjects, seeds and prediction lengths
match their metadata. The 78 Keras model packages also pass ZIP integrity checks;
one additional HDF5 weight file is included and checksummed.

All six recordings load through the active research preprocessing code with eight
active trials in each session. Re-importing the normalization and 60-epoch
retest-map results in an isolated temporary tree reproduces all 1,608 supplementary
dashboard score rows exactly, with identical packed targets and predictions.
The existing React TypeScript check and production build pass. No training run was
needed for these transfer checks.
