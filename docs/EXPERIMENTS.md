# Running experiments and updating the dashboard

An experiment is one JSON file, `EXPERIMENTS/configs/<id>.json`. It says what changes relative to the training
baseline and how the experiment appears on the dashboard. Everything else (training, scoring, the dashboard page) is
derived from it.

## The spec

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Letters, digits, `_ . + -`. Must equal the file name. Never reuse an id for a different setup |
| `title` | yes | Short human title |
| `changes` | yes | Settings that differ from the baseline (`BASE` in `EXPERIMENTS/pipeline/run.py`, table below) |
| `description` | no | What the experiment tests; falls back to the older field `hypothesis` |
| `tags` | no | List of strings, e.g. `["active torque", "dropout"]` |
| `results_from` | no | Reuse another experiment's saved predictions (scoring-only variants) |
| `dashboard` | no | Present = shown on the dashboard. `{"group", "label", "id"?, "name"?, "order"?, "blurb"?, "text"?}`; usually written by `dashboard tag` |

`dashboard.group` must be a group in `EXPERIMENTS/dashboard_groups.json` (Models,
Normalization study, Exploration). `dashboard.text` may hold `purpose`, `target`, `protocol`, `overview`, `params` and
`report` (a repository-relative Markdown file shown on the experiment page).

### Baseline settings (`changes` overrides these)

| Key | Baseline | Common values |
| --- | --- | --- |
| `passive` | `legacy` | `static` (each session's relaxed holds; the A0 default), `none` (total torque), `legacy` (old automatic rule), others in `ML/research/passive.py` |
| `emg` | `trainmax` | `trainmax` (÷ session-1 maximum), `train_p99`, `session_p99`, `mvc_global`, `mvc_angle`, `agc_auto` (`ML/research/emg_norm.py`) |
| `window` | `20` | samples at 100 Hz (20 = 0.2 s) |
| `model` | `lstm` | `lstm` (uses `units`, `layers`, `dropout`), `ridge`, `moe` (`ML/research/models.py`) |
| `units`, `layers`, `dropout` | `8`, `2`, `0.3` | LSTM size |
| `balance` | `none` | `scale` (each angle's loss scaled by its swing), `weights` |
| `lr` | `null` | learning rate; `null` keeps Nadam 0.003 |
| `train_protocol` | `orig` | `orig` (batch 8, ≤200 epochs, early stop 25) or `fast` |
| `shuffle` | `true` | shuffle training windows |
| `pf_extra`, `rest_weight`, `extra_input` | `null` | rarely used extras (see `ML/research/train.py` and `pipeline/run.py`) |

There is no inheritance between specs: to build on A0, copy A0's `changes` and edit them.

### Example

```json
{
  "id": "TEAM01_active_dropout",
  "title": "A0 with dropout 0.2",
  "description": "Does less dropout help the active-torque model at the plantarflexed angles?",
  "tags": ["active torque", "dropout"],
  "changes": {"passive": "static", "emg": "trainmax", "balance": "scale", "window": 20, "dropout": 0.2}
}
```

## Workflow

Run every command from the repository root. On Windows use `.\.venv\Scripts\python.exe` for `.venv/bin/python`.

**1. Write the spec** as above. `python -c "from EXPERIMENTS.pipeline.spec import load; print(load('TEAM01_active_dropout'))"`
checks it.

**2. Run one job** (one person, one direction, one seed). This is a full training run, not a quick test:

```bash
.venv/bin/python -m EXPERIMENTS.pipeline.run --exp TEAM01_active_dropout --subject EG --direction S1toS2 --seed 0
```

It writes `EXPERIMENTS/results/TEAM01_active_dropout/EG/S1toS2/seed0.npz` and `.json`. Add `--save-weights` to keep
the trained weights.

**3. Run the full set** in parallel. The queue skips jobs that already have results *with the same config*; if you
changed a spec after running it, the queue stops with `ConfigMismatch` instead of mixing old and new results. Give a
changed setup a new id.

```bash
.venv/bin/python -m EXPERIMENTS.pipeline.run_queue --exps TEAM01_active_dropout \
    --subjects YES JM EG IES CH HM --directions S1toS2 --seeds 0 1 2 --workers 6
```

Each job's log goes to `EXPERIMENTS/logs/`. `QUEUE_DONE` means the queue finished, not that every job succeeded: look
for `FAILED` lines.

**4. See what is there:**

```bash
.venv/bin/python -m EXPERIMENTS.pipeline.dashboard list
```

lists every spec with its dashboard place, tags and how many results exist per person.

**5. Name it and place it** on the dashboard. This writes the `dashboard` block (and the description and tags) into the
spec; `--dry-run` shows the result without writing.

```bash
.venv/bin/python -m EXPERIMENTS.pipeline.dashboard tag TEAM01_active_dropout --label "Dropout 0.2" --group Models \
    --name "A0 with dropout 0.2" --description "A0 with dropout 0.2 instead of 0.3" --tag dropout --order 20 \
    --blurb "Same as A0, with less dropout."
```

The same command **renames or re-describes** an existing entry: run it again with the new label, name, description or
tags (`--tag` replaces the existing tags).

`export` refuses to mix runs: if any saved run of an experiment was made with a different config from the others, it
stops and names the file.

**6. Score and rebuild:**

```bash
.venv/bin/python -m EXPERIMENTS.pipeline.dashboard build TEAM01_active_dropout
```

`build` = `export` (score every position of every saved run, add the error breakdown, pack the waveforms into
`EXPERIMENTS/dashboard_react/src/data/presentation-catalog.json`) followed by `npm run build`, which writes
`EXPERIMENTS/dashboard_react/dist/index.html`. With no ids it re-exports every tagged experiment. `export` alone
skips the React build.

**7. Take an entry off the dashboard** (its results and spec stay):

```bash
.venv/bin/python -m EXPERIMENTS.pipeline.dashboard remove TEAM01_active_dropout
```

### Commit checklist

Commit these together:
- the spec (`EXPERIMENTS/configs/<id>.json`)
- the results you want to share (`EXPERIMENTS/results/<id>/…`); then run
  `python EXPERIMENTS/code/verify_handoff.py --write-manifest` and include `EXPERIMENTS/handoff_manifest.json`
- `presentation-catalog.json` and `dist/index.html`
- any report text the spec's `dashboard.text.report` points to

Then check the run with `python -m EXPERIMENTS.analysis.verify_study` (window study) or by opening the dashboard.

## Adding a sidebar group

Groups, their captions and colours, and the landing page live in `EXPERIMENTS/dashboard_groups.json`. Add a group
there before tagging into it; `export` refuses an unknown group. `extra_items` places entries that have no spec (last
semester's model, the `main` session-passive model, a few explorations).
