# Weekly update, 27 Oct 2026

Sources for the presentation and the analyses behind it. The live deck is the claude.ai artifact
<https://claude.ai/artifact/Q1pbcnDAu5KB6N1LFzKT3G> (private until shared; export to PDF or PowerPoint from its Share menu).
This folder is a copy so nothing depends on that page.

## Deck (`deck/`)

`deck.json` (slide order and sections) and one HTML file per slide in the Slides artifact format. Images are referenced by
artifact asset id; the matching files are in `figures/`:

| Slide | Asset id | File |
| --- | --- | --- |
| data | 155a78d5ba91d7783edcd46a114e8e71 | fig_data.png (HM, session 1, −0.36 rad, one 0.5 s window) |
| target | cdc2192781ef9545e67c228cec0ed86f | fig_passive_map.png (EG passive maps, both sessions) |
| retest | dbf55b531ac1b8db10032ea14be5dc55 | fig_target.png (EG session 2, −0.04 rad: measured, zoom, target) |
| flow | 28c21c1d09f26a68cc17caae4a849901 | pipeline_tree.png (the whole pipeline for EG at −0.04 rad, both sessions, seed-0 model) |
| autogain | 567bb535b4873706593846d7eb67bb2b | agc_timeseries.png (JM −0.64 rad, from the 6 Oct supervisor deck) |
| autogain-effect | ca5e57327dc625e85c28a52fb5aaa61e | emg_size_change_before_after_agc.png (same source) |
| balance | 63dd900e8da13619f86680b74827c5d0 | fig_balance.png (YES share of the training loss per angle) |
| dashboard | 368c47a13a91d5251bb83dabc3a71fbb | dashboard.png (screenshot of the results explorer) |

The auto-gain idea is from Tanaka T, Nambu I, Maruyama Y, Wada Y (2022), *Sensors* 22(13):5005, doi:10.3390/s22135005.

## Scripts (`scripts/`)

Run from the repository root with the project interpreter (`.venv311`), `PYTHONPATH` set to the root.
The data folder is `EMG_TQ_Modelling_data` next to the repository (recordings plus the YES worksheet PDF and
`experiment_metadata_IES.docx`, kept out of git because they hold personal details).

| Script | What it does |
| --- | --- |
| `make_figures.py <data> ML/reviewed_overrides.json <emg_agc.py> <out>` | Deck figures from the recordings (`emg_agc.py` is in the `experimentation-dorsiflexion` branch, `ML/preprocessing/`). Also writes `fig_agc_how.png`, no longer used. |
| `pipeline_tree.py <data> ML/reviewed_overrides.json <runs>/window50_retest_map/EG_seed0 <out.html>` | The pipeline tree (slide 8) from real EG signals, maps and predictions. Render: `msedge --headless=new --force-device-scale-factor=2.5 --window-size=1664,700 --screenshot=pipeline_tree.png file:///<out.html>`. |
| `check_passive_maps.py <overrides> <subjects…>` | Loads each person through the 0.5 s + retest map pipeline; prints windows, passive maps, positions. |
| `scan_retest.py <data> <overrides>` | For every session-2 trial: relaxed torque vs the session-2 and session-1 passive maps. |
| `probe_agc.py`, `agc_summary.py` | Session-2 / session-1 calf EMG size per Nm of push, before and after auto-gain. |
| `consistency.py <dashboard dir>`, `consistency_ablation.py <dashboard> <results> <runs>` | Seed-to-seed spread of each model; single changes paired against the original model. |
| `compare_budget.py <60-epoch runs> <150-epoch runs> <dashboard>` | The longer-training check below. |

## Findings recorded here

- **Retest map:** the session-2 passive map is not always closer to session 2’s relaxed level than the session-1 map;
  it depends on person and angle (e.g. YES −0.49 rad sits below both). Mentioned in the slide-6 speaker notes.
- **Auto-gain on the recordings** (5 people, CH has it off): cases where session-2 calf EMG size per Nm is off by more than
  25 % fall from 32 % to 18 % (all angles) and from 53 % to 30 % (toes-down angles); 72 % move closer to session 1.
- **Seed consistency** (median within-cell SD of MAE across seeds, session 1 → 2): new model D01 0.058 Nm vs
  0.5 s + retest map 0.111 Nm on CH, HM, IES, JM (10 seeds). Equal weighting alone ≈ 0.6× and auto-gain alone ≈ 0.7×
  the original model’s spread; window length does not matter (D02 ≈ D01).
- **Longer training check** (0.5 s + retest map, 6 people × seeds 0–2, 150-epoch cap / patience 25): spread 0.114 → 0.098 Nm,
  accuracy unchanged (MAE 0.88 Nm). The epoch cap is not the main reason D01 is steadier. Runs:
  `EMG_TQ_Modelling_runs/window50_retest_map_150ep`; the main 60 runs are in `EMG_TQ_Modelling_runs/window50_retest_map`.
