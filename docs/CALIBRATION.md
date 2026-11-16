# Target and the error breakdown

## The default target: active torque with each session's own passive torque

The model predicts **active torque**: measured torque minus the passive torque of the ankle at that angle. Before
every active trial the person sits relaxed for about 30 s at the same angle. Those relaxed holds give one passive table
per session, and each session's active trials use **their own session's** table (`"passive": "static"` in a spec).

This uses session-2 data, but only from relaxed recordings made before the active trials, never the torque being
predicted. It is part of the default model, not a calibration. A0 (`EXPERIMENTS/configs/A0_active_trainmax.json`) is
this default and the dashboard opens on it.

- **IES** recorded no relaxed holds and the old automatic rule finds no passive trials either, so in A0 and all earlier
  experiments nothing is subtracted for IES: its target is total torque. From the window study (13 Nov 2026) on,
  `"metadata_passive": true` subtracts the passive torques noted at each position during IES's MVC trials
  (`ML/subjects/experiment_metadata_IES.docx`, one table for both sessions; `METADATA_PASSIVE` in `ML/config.py`).
- `"passive": "none"` predicts total torque instead (the N/NR normalization study).

## No calibration

Predictions are always scored as saved, on the whole trial. An opt-in 10 s gain + offset calibration (fitted on the
first 10 s of each test trial's measured torque, scored from 30 s) existed until 16 Nov 2026. No current result used
it, so it was removed: a spec with a `calibration` field is now rejected (`EXPERIMENTS/pipeline/spec.py`). Its earlier
results are described in `EXPERIMENTS/reports/normalization/REPORT.md`.

## The error breakdown (every entry)

The error breakdown **describes** error; it never changes a score. For every scored trial,
`EXPERIMENTS/pipeline/scoring.py` stores two extra numbers, both over 30 s → end of trial:

| Field | Meaning |
| --- | --- |
| `err_raw_mae` | MAE of the prediction as saved |
| `err_after_gain_offset_mae` | MAE after a gain + offset fitted on the first 10 s of the trial (gain clipped to [0.5, 2]; offset only if the torque barely moves) |

The difference is the part of the error that is **drift** (wrong scale or offset); what remains is **shape** error.
The experiment page shows this line under the scores. The model's own score is unchanged.
