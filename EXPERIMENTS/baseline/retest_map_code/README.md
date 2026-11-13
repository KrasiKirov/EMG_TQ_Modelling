# Source of the displayed retest-map benchmark

`ML/` is copied without edits from
`EXPERIMENTS/run_collections/window50_retest_map/EG_seed0/source/`, with that run's
`reviewed_overrides.json` next to it. The recorded source revision is
`fa5f949ee88842565428f706502f27faa4b9bcc2`. Each original run also retains its own
source snapshot and provenance. The 150-epoch collection has its own recorded
revision and should remain a separate experiment.

This copy supplies the `ML.training.parallel_benchmark` entry point referred to
by the dashboard's historical README. That module is absent from the active `ML/`
package on `experimentation-normalization`. Keeping the historical implementation
here avoids replacing the normalization framework with a different pipeline.

To inspect or run this implementation, use this directory as the Python working
directory and `PYTHONPATH`, so its `ML` package is imported. Pass absolute paths to
the repository's `ML/subjects`, this copy's `ML/reviewed_overrides.json`, and a new
output directory. Use `--help` before launching training. Do not point a new run
at one of the preserved collections: the historical runner can remove interrupted
job folders when resuming.

See [the handoff guide](../../HANDOFF.md) for setup and pipeline cleanup priorities.
