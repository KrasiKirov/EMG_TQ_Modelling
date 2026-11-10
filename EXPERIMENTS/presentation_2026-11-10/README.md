# Presentation, 10 Nov 2026: two improvements over last semester's LSTM

Live deck: <https://claude.ai/artifact/HhEzHdcGnSn7edegmDCzFe> (private until shared; export to PowerPoint/PDF from its Share menu). `deck/` is a copy.

Numbers: `ladder.py` -> `ladder_runs.csv`, `ladder_groups.csv` (spreads: `spread.py`). Every model is rescored from the dashboard
waveforms (0.2 s grid) on the same key (active torque, clean passive table), seeds 0-2, 6 people, S1 -> S2, scored from 30 s
to trial end, 10 s gain+offset calibration as in `normalization/calibration_test.py`. A0 rescored on the grid = 0.608 Nm toes-down
vs 0.609 at full resolution.

| Toes-down MAE (Nm) | value | vs baseline |
| --- | --- | --- |
| Last semester (X00f, framework re-run) | 0.852 | |
| Balanced positions (A0) | 0.608 | -29 % |
| Balanced + 10 s calibration (A0+cal) | 0.434 | -49 % |

Other angles: 0.949 -> 0.759 (-20 %) -> 0.640 (-33 %). Original code X00 (seeds 0-9): 0.857 toes-down.
Figures: `fig_calibration.py` (YES, -0.381 rad, seed 0 of A0), `fig_balance.png` (copied from the 3 Nov deck).
