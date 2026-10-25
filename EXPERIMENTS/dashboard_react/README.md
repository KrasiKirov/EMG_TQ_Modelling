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
