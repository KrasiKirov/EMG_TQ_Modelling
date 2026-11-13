# Legacy catalogue scripts

These rebuild the historical parts of the dashboard data (the family and research studies in
`src/data/catalog.json`, and the focused `presentation-catalog.json` they feed). They are kept so those
historical entries can be regenerated, but they are **not** part of the normal workflow.

To put an experiment on the dashboard, tag it and export it:

```bash
python -m EXPERIMENTS.pipeline.dashboard tag <EXP> --label "..." --group Models
python -m EXPERIMENTS.pipeline.dashboard build
```

`build_presentation_catalog.py` replaces the whole focused catalogue. After running it, run
`python -m EXPERIMENTS.pipeline.dashboard export` to restore every tagged experiment.
