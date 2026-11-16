# Data

## Subjects and files

Six people, one `.flb` file each in `ML/subjects/`. Each file holds both sessions. The registry is `SUBJECTS` in
`ML/config.py`; set `EMG_DATA_DIR` to read the files from somewhere else.

| Subject | File | Channel profile | Notes |
| --- | --- | --- | --- |
| IES | `IES_081024.flb` | `IES01` (non-standard channel order) | No relaxed holds or usable passive trials (target = total torque unless `metadata_passive`); passive torques per position in `experiment_metadata_IES.docx`. `main` used `default` (open question) |
| YES | `YES_281124.flb` | `default` | Passive values also hand-entered in `reklab_trial_worksheet_YES_20241128.pdf`; some trials mislabelled, fixed by overrides |
| EG | `EG_121224.flb` | `default` | |
| CH | `CH_151224.flb` | `default` | |
| JM | `JM_140325.flb` | `default` | |
| HM | `HM_110425.flb` | `IES01` | Same channel order as IES. `main` used `default` (open question, see `PIPELINE_COMPARISON.md`) |

Trial and passive-torque corrections for individual recordings are in `TRIAL_OVERRIDES` and `PASSIVE_TORQUE_OVERRIDES`
in `ML/config.py`.

## Sessions, positions and trial types

- **Sessions.** Session 1 (the lab calls it "test") and session 2 ("retest"), both in the same file. A
  trial's session comes from its comment: `retest` → session 2, `test` or `repeat` → session 1.
- **Positions.** Eight fixed ankle angles (`p1` … `p8` in trial comments), spread over the range of motion. Each
  session has one active trial per angle. The dashboard calls the two most plantarflexed angles "plantarflexed".
- **Trial types** (`ML/preprocessing/dataset_builder.py`, from the comment):
  - `mvc…` → maximum voluntary contraction
  - `…passive…` → passive trial: the ankle is moved or held while the person is relaxed
  - anything else → active isometric trial: the person varies their torque at a fixed angle. Trials whose torque SD
    exceeds 30 Nm are passive movement or ramp trials and are left out of training and of the passive lookup.
- **Relaxed holds.** Before each active trial the person rests about 30 s at the same angle. `ML/research/data.py`
  finds these holds in every non-active, non-MVC trial (angle SD < 5 mrad, torque SD < 1 Nm, at least 5 s) and attaches
  each to the next active trial at the same angle (within 0.05 rad).

## Signals

Recorded at 1 kHz: ankle angle (rad), measured torque (Nm) and four EMG channels in the order GM, GL, SOL, TA
(gastrocnemius medialis, gastrocnemius lateralis, soleus, tibialis anterior). The experiment pipeline turns each EMG
channel into an envelope (band-pass 30–300 Hz, rectify, 2 Hz low-pass), trims the first 1 s of each trial (filter
start-up) and down-samples to 100 Hz. Model inputs are 20-sample (0.2 s) windows of the four envelopes plus the angle.

## Total and active torque

- **Total torque** is what the torque sensor measures.
- **Passive torque** is the torque the ankle shows when the person is relaxed at that angle.
- **Active (voluntary) torque** = total − passive. It is the default target. How passive torque is estimated, and why
  using session 2's own relaxed holds is not a calibration, is in [`CALIBRATION.md`](CALIBRATION.md).

## The `.flb` format

`ML/preprocessing/flb_reader.py` reads REKLAB `.flb` files (format reverse-engineered from the lab's `flbio.m` /
`flb2mat.m`). A file is a sequence of trial blocks. Each block has an integer header (format version 2–4, number of
channels, samples per channel, sample format int16 or float32), the sampling interval and start, the trial comment,
the channel names, per-channel min and max for scaling, then the samples. Each trial becomes one pandas DataFrame with
the comment in its attributes. To look at a file:

```bash
.venv/bin/python ML/explore.py --help
```

## Saved predictions

Each run writes `EXPERIMENTS/results/<id>/<subject>/<direction>/seed<k>.npz` with `y_own_target` (the run's own
target), `y_pred`, `trial`, `angle`, `session` and `t`, plus a `.json` with the resolved config, window counts,
training info and timings. Scores are always recomputed from these arrays, never from the dashboard's sampled traces.

## Checking the transferred artifacts

`EXPERIMENTS/handoff_manifest.json` lists the path, size and SHA-256 of every recording, saved prediction and saved
model shipped with the repository. Check them with the standard library only:

```bash
python EXPERIMENTS/code/verify_handoff.py           # sizes and checksums
python EXPERIMENTS/code/verify_handoff.py --quick   # paths and sizes only
```

After deliberately adding or changing artifacts, run it with `--write-manifest`, review the diff, and commit the
manifest with the change.
