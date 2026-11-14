# Target, calibration and the error breakdown

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

## The 10 s gain + offset calibration (opt-in)

A calibration experiment corrects every test trial with a gain and an offset fitted on the **first 10 s of that same
trial's measured session-2 torque** (active torque for A0cal, total torque for N0cal):

    p' = a · p + b      fitted on 0–10 s, a clipped to [0.5, 2]
                        if the measured torque barely moves (SD < 0.2 Nm), only the offset is fitted

and is scored from **30 s to the end** of the trial. It needs the person to push while torque is measured, which a
real deployment may not have, so it is never the default. It lives in the "Calibration experiments" group of the
dashboard (`N0cal`, `A0cal`).

Turn it on in a spec. A scoring-only variant can reuse another experiment's predictions with `results_from`:

```json
{
  "id": "A0cal_active_trainmax",
  "title": "A0 + 10 s calibration",
  "results_from": "A0_active_trainmax",
  "changes": {"passive": "static", "emg": "trainmax", "balance": "scale", "window": 20},
  "calibration": {"type": "gain_offset", "seconds": 10, "score_from_s": 30}
}
```

`calibration` defaults to `{"type": "none"}`. `seconds` must be positive and `score_from_s` must be at least
`seconds`, so the scored part never overlaps the part used for fitting (`EXPERIMENTS/pipeline/spec.py`).

## The error breakdown (every entry)

Calibration is also used to **describe** error, not to change a score. For every scored trial,
`EXPERIMENTS/pipeline/scoring.py` stores two extra numbers, both over 30 s → end of trial:

| Field | Meaning |
| --- | --- |
| `err_raw_mae` | MAE of the prediction as saved |
| `err_after_gain_offset_mae` | MAE after the 10 s gain + offset above |

The difference is the part of the error that is **drift** (wrong scale or offset); what remains is **shape** error.
The experiment page shows this line under the scores. The model's own score is unchanged.

## Comparing entries

Uncalibrated entries are scored on the whole trial; calibrated entries only from 30 s. They are different protocols,
so compare a calibrated entry only with another calibrated one, or compare the `err_*` fields, which use the same
window for every entry.
