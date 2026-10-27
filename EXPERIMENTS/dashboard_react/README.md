# EMG → ankle torque · results explorer (React)

An interactive page for the capstone results: pick two models (A and B) by their settings and compare
them per person, per ankle position and per training run. It replaces the single-file HTML dashboard in
`EXPERIMENTS/dashboard/` with a cleaner interface built with Vite, React 18, TypeScript, Mantine (UI)
and Apache ECharts (charts).

The build produces **one self-contained file**, `dist/index.html` (about 4.6 MB, all data, code and styles inlined).
Double-click it to open it in any modern browser on Windows or macOS. It needs no server, no install
and no internet. Share that one file.

## What is inside

| Tab | Shows |
| --- | --- |
| Overview | KPI cards (median at the toes-down positions and at the other 6, B − A, 5–95 % and 1–99 % ranges), box plots per position with every run as a dot, and a B − A heat map over every person and position (click a cell to open its waveforms) |
| Waveforms | Measured voluntary torque and the predictions of A and B for one person, position and run (typical = median-R² run, best, worst or any seed), zoomable; table of every run at that position |
| Training | Learning curves (fit / validation / test MAE per epoch, ours vs last semester), a summary at the kept epoch, and where the error comes from (offset, other systematic, run-to-run) with run-average (ensemble) scores |
| Experiments | All 15 experiments vs the last-semester reference on matched runs (click a row to load it as model B), the earlier round-2 exploration and the report-1 network types |
| About | Definitions, display rules, how to refresh the data, hardware |

Display rules (from the supervisor): R² below 0 is shown as 0 by default (switch in the left panel),
medians over runs, spread as 5th–95th and 1st–99th percentiles (numpy linear interpolation), MAE in Nm
next to the peak voluntary torque of the position.

Conventions used by the explorer:

- R² floor: each run's R² is set to 0 first, then medians / percentiles are taken (as in the deck's box
  plots and headline numbers). The appendix tables take the percentile of the raw values first, so a few
  tail percentiles can differ slightly.
- B − A is computed on the people for whom both models were trained; the KPI cards name them when one
  model is missing for someone.
- Better / worse colours only from 0.02 R², or 0.05 Nm and 2 % of the peak torque for MAE; smaller
  differences are shown as "about the same".
- Bias is compared by its size (|bias|); the box plots keep the sign.
- Values far outside the bulk (raw R² below −3, one-off MAE / bias outliers) are drawn at the chart edge
  with ▼ / ▲ markers and the true median as a label.

The view is kept in the URL hash, so a link opens the same view, e.g.
`dist/index.html#tab=waveforms&subj=JM&dir=S1toS2&metric=mae&pos=-0.787&theme=dark`
(keys: `tab`, `subj`, `dir`, `metric`, `floor`, `a`, `b`, `wsubj`, `pos`, `ra`, `rb`, `theme`).

## Use it

Requires Node.js 18 or newer (tested with Node 24, npm 11).

```bash
cd EXPERIMENTS/dashboard_react
npm install          # once
npm run dev          # development server with hot reload (prints a localhost URL)
npm run build        # type check (tsc --noEmit) + build -> dist/index.html
```

`npm run build` must finish without TypeScript errors; it then writes `dist/index.html`.

## Refresh the data

The data file `src/data/results.json` is generated; never edit it by hand. From the repository root, once the
training jobs have finished, run the steps in this order (each reads the previous step's output):

```bash
python EXPERIMENTS/code/presentation_metrics.py  # scores every saved run -> results/presentation_runs.csv
python EXPERIMENTS/code/deck_figures.py          # bias / variance split  -> results/bias_variance_runs.csv
python EXPERIMENTS/code/build_dashboard.py       # rewrites src/data/results.json (and the old HTML dashboard)
cd EXPERIMENTS/dashboard_react
npm run build                                    # new dist/index.html
```

The run counts shown in the Training tab (e.g. "10 runs per person") come from the data, so nothing in the
code needs changing when more seeds are added.

## Code layout

```
src/
  lib/            pure TypeScript, no React: data access, percentiles, resolution rule,
                  aggregation for KPIs / heat map / tables, waveform decoding, URL hash
  charts/         ECharts option builders
  components/     sidebar (model configurators, view controls), shared UI, chart wrapper
  tabs/           one file per tab
  data/results.json
```

How a model is resolved: for a person and direction, the explorer looks for an experiment with exactly
the chosen settings that was trained for that person. If there is none and the person is not JM, it
also accepts an experiment that differs only in the passive table (the original and clean tables are
identical for everyone except JM), e.g. C01 stands in for D01 for YES and EG. Otherwise it says the
combination was not trained (the original code only ran learn 1 → test 2).

For automated checks, numbers are also in the DOM with `data-testid` attributes: `kpi-toes-A`,
`kpi-toes-B`, `kpi-toes-delta`, `kpi-other-*`, `heat-<PERSON>-<k>`, `status-A`, `status-B`, and table rows
(`exp-row-<id>`, `runs-row-<A|B>-<seed>`, `lc-row-*`, `bv-row-*`, `r2-row-*`, `fam-row-*`), e.g.
`chrome --headless=new --virtual-time-budget=10000 --dump-dom "file:///…/dist/index.html#tab=overview"`.


## Experiment-first workflow (25 October 2026)

The default Experiment tab now uses a single saved experiment selected from the left-hand list. It shows the experiment purpose, implementation-based explanations (including an interactive AGC example), all-subject summaries, all six saved performance metrics by position, individual runs, and one experiment’s prediction against the measured target. Train/test direction is selectable. The selected experiment, subject, position and run are stored in the URL hash (`exp`, `wsubj`, `pos`, `ra`).

Comparison is a separate workspace with its own A/B presets and Summary, Waveforms and Training audit views. Study history explains the sequence of trials and the recorded decisions. Its interpretations incorporate the 24 October audit corrections; the selected candidate is not labelled a confirmed best model.

Exact experiment runs are preferred. The existing passive-table equivalence rule is optional and every substitute source is named in the subject table; missing results remain missing. CSV export contains raw run-level scores and both selected and source experiment IDs. Single-experiment summaries take a median across runs at each subject-position, then a median across the chosen positions. Signed bias is preserved, so summary offsets can cancel. Raw individual-run scores are always available regardless of the R² display floor. Training curves are separately recorded diagnostic runs and only shown for an exact experiment/subject/direction match.

New UI files: `tabs/ExperimentTab.tsx`, `tabs/ExperimentLibrary.tsx`, `components/ExperimentSidebar.tsx`, `components/ExperimentMethod.tsx`, and `lib/experimentStory.ts`. No recordings, predictions, training settings or generated results were modified.


## Extended catalogue, timing and demeaning

`src/data/catalog.json` supplements the original generated dataset. Rebuild it from existing saved predictions (no training) with `python scripts/build_catalog.py`, using the project Python environment with numpy, pandas and matplotlib. It exports five family models, 38 earlier study/configuration collections, main-run timing metadata and full-trial means. Three teammate entries are transcribed directly from the local `origin/main` Markdown reports, with revision and native aggregation retained. They intentionally do not invent unavailable waveform, seed, or timing records.

The earlier family targets differ from the clean passive-trial evaluation target; this is labelled explicitly. Study collections remain separate, including the two ridge variants. Full float32 waveform storage preserves extreme ridge failures without int16 clipping. Box plots display quartiles, a median and 5th–95th percentile whiskers plus individual seeds. Timing is wall-clock per saved run, with the measured scope labelled (family fit-only versus research/control fit and prediction).

The new waveform switch removes each curve’s own full-trial mean for evaluation only. The main controlled models do not remove the torque mean in training. Ridge fits an intercept using a weighted training mean; the 0.5 s + retest map benchmark globally centres/scales its neural-network target on training-only statistics and undoes that at prediction time. Those training transformations are documented separately from per-trial demeaned scoring.


## Focused interface for the supervisor presentation

The shipped client uses `presentation-catalog.json`: all 15 controlled experiments, the GRU family run and the 0.5 s window + retest passive map runs. The earlier research collections are retained on disk in the archived catalogue but excluded from the interface and accepted experiment links. The left navigation follows 0.5 s + retest map / new model / preliminary exploration. Controlled experiments has its own complete tab. Each saved experiment has Results, Waveforms and How it works views. The result view leads with figures and tables; metric definitions, provenance and timing detail expand on demand.

Box plots now offer individual-axis zoom or a shared scale, plus ordinal position focus. Individual-axis limits include all displayed points and explicitly show the changed axis range; they do not trim outliers. The focused catalogue retains only the archived GRU waveform payload to reduce startup cost.

Refresh the focused supplement after rebuilding the archived catalogue with `python scripts/build_presentation_catalog.py`, then run the normal dashboard build. Only the explicitly requested collections are shipped in the focused interface.


## 0.5 s window + retest passive map

`main/session-passive` holds 60 runs of the 0.5 s LSTM (6 people × seeds 0–9, session 1 → 2): 50 samples of the four muscle envelopes plus angle, LSTM 8×2, up to 60 epochs, patience 12, batch 64, train stride 5. The target is measured torque minus passive torque; session 1 uses its own passive map and session 2 uses a passive map re-measured in session 2.

Data fixes in the pipeline (`ML/reviewed_overrides.json`, `ML/config.py`, `ML/preprocessing/pipeline.py`): YES trials 27/29/31/33 are the retest pushes at p1–p4 (their recorded notes wrongly say “passive”), and YES session-1 passive values come from the lab worksheet; IES has no passive trials, so its documented pre-test table (`experiment_metadata_IES.docx`) is used for both sessions; CH is added to the subject list. JM needs no fix: the still-hold rule already ignores the movement in passive trial 21.

Train (from the repository root; the data folder holds the `.flb` recordings, the YES worksheet PDF and the IES DOCX, none of which are in git):

```bash
python -m ML.training.parallel_benchmark --output <runs folder> --data-dir <data folder> --overrides ML/reviewed_overrides.json --workers 8
```

Then import with `python scripts/import_retest_map.py <runs folder>` (requires NumPy) and run the normal build. The importer only reads arrays and JSON, recomputes every position score in float64 and checks it against each run’s `metrics.json`. Full trial means are preserved for evaluation demeaning. Chart resampling never extrapolates beyond valid intervals, and gaps remain gaps. `build_presentation_catalog.py` preserves the imported collection on later refreshes.

Comparison now offers Saved experiments (0.5 s + retest map, GRU and all controlled models) plus the original Controlled settings workspace. Saved comparisons match available angles within 0.02 rad, retain each protocol’s own target/interval, and do not imply paired seeds or a matched improvement. Each waveform panel has its own target; both panels share time and torque scales. The comparison box plot uses each seed’s median across common angles, while the table takes medians across seeds within an angle first.

Training times are labelled per seed. The 0.5 s + retest map timings include model construction, compilation and fit, with 8 single-thread jobs sharing the CPU; controlled timings include prediction. These hardware and timer differences preclude a controlled speed claim. Its learning histories are shown in standardized MSE units, separately from the controlled diagnostic curves.
