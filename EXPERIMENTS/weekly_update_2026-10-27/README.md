# Weekly update, 27 Oct 2026

Sources for the presentation and the analyses behind it. The live deck is the claude.ai artifact
<https://claude.ai/artifact/Q1pbcnDAu5KB6N1LFzKT3G> (private until shared; export to PDF or PowerPoint from its Share menu).
This folder is a copy so nothing depends on that page.

## Deck (`deck/`)

`deck.json` (slide order and sections) and one HTML file per slide in the Slides artifact format. Images are referenced by
artifact asset id; the matching files are in `figures/`:

The deck is 9 slides: cover, agenda, `flow` (last week’s method), `changes`, `newflow` and `agczoom` (new method),
`balance`, `achieve`, `dashboard`.

| Slide | Asset id | File |
| --- | --- | --- |
| flow | 28c21c1d09f26a68cc17caae4a849901 | pipeline_tree.png (last week’s pipeline for EG at −0.04 rad, both sessions, seed-0 model) |
| newflow | d113d4a4c1922c116fd65903c790a566 | pipeline_tree_new.png (same tree for the new method; EG’s D01 = C01, seed 0) |
| agczoom | ab4f60c2a4bd3d522fcbfc266b49a361 | autogain_zoom.png (JM soleus, −0.36 rad: before, running levels, rescale, after) |
| balance | 63dd900e8da13619f86680b74827c5d0 | fig_balance.png (YES share of the training loss per angle) |
| dashboard | 368c47a13a91d5251bb83dabc3a71fbb | dashboard.png (screenshot of the results explorer) |

Removed on 27 Oct (in git history at 7fac500): data, pipeline, target, retest, evaluation, autogain and
autogain-effect, whose figures (fig_data, fig_passive_map, fig_target, agc_timeseries,
emg_size_change_before_after_agc) are still in `figures/`.

The auto-gain idea is from Tanaka T, Nambu I, Maruyama Y, Wada Y (2022), *Sensors* 22(13):5005, doi:10.3390/s22135005.

## Scripts (`scripts/`)

Run from the repository root with the project interpreter (`.venv311`), `PYTHONPATH` set to the root.
The data folder is `EMG_TQ_Modelling_data` next to the repository (recordings plus the YES worksheet PDF and
`experiment_metadata_IES.docx`, kept out of git because they hold personal details).

| Script | What it does |
| --- | --- |
| `make_figures.py <data> ML/reviewed_overrides.json <emg_agc.py> <out>` | Deck figures from the recordings (`emg_agc.py` is in the `experimentation-dorsiflexion` branch, `ML/preprocessing/`). Also writes `fig_agc_how.png`, no longer used. |
| `pipeline_tree.py <data> ML/reviewed_overrides.json <runs>/window50_retest_map/EG_seed0 <out.html> [new_method_EG.npz]` | The pipeline tree from real EG signals, maps and predictions; with the .npz, the new-method tree. Render: `msedge --headless=new --force-device-scale-factor=2.5 --window-size=1664,700 --screenshot=<png> file:///<out.html>`. |
| `new_method_data.py <dorsiflexion worktree> <results/C01_autogain+equalweight or -> <out.npz> [subject] [angle]` | Runs the new method’s own code (`ML.research`, only in the `claude/dorsiflexion-r2-drop-341137` worktree) and saves auto-gained inputs, running floor/top, passive maps, targets and predictions. Used as `EG -0.04` and `JM -0.36`. |
| `autogain_zoom.py <new_method_JM.npz> <out.html>` | The auto-gain zoom (four steps, JM soleus); prints the before/after size ratios and the worked example. |
| `check_passive_maps.py <overrides> <subjects…>` | Loads each person through the 0.5 s + retest map pipeline; prints windows, passive maps, positions. |
| `scan_retest.py <data> <overrides>` | For every session-2 trial: relaxed torque vs the session-2 and session-1 passive maps. |
| `probe_agc.py`, `agc_summary.py` | Session-2 / session-1 calf EMG size per Nm of push, before and after auto-gain. |
| `consistency.py <dashboard dir>`, `consistency_ablation.py <dashboard> <results> <runs>` | Seed-to-seed spread of each model; single changes paired against the original model. |
| `compare_budget.py <60-epoch runs> <150-epoch runs> <dashboard>` | The longer-training check below. |

## Findings recorded here

- **Retest map:** the session-2 passive map is not always closer to session 2’s relaxed level than the session-1 map;
  it depends on person and angle (e.g. YES −0.49 rad sits below both). Was in the removed retest slide’s notes.
- **Example angle, new vs last week** (EG −0.04 rad, seed 0): R² 0.986 / MAE 0.84 Nm vs 0.983 / 1.08 Nm. Both methods
  give EG the same passive maps (−3.49 Nm session 1, −2.60 Nm session 2 at this angle).
- **Auto-gain zoom** (JM −0.36 rad, session-2 / session-1 EMG size per Nm): GM 0.36 → 0.91, GL 0.83 → 1.04,
  SOL 0.38 → 0.98. At EG −0.04 rad the sizes were already close (0.93–1.21), so auto-gain changes little there.
- **Auto-gain on the recordings** (5 people, CH has it off): cases where session-2 calf EMG size per Nm is off by more than
  25 % fall from 32 % to 18 % (all angles) and from 53 % to 30 % (toes-down angles); 72 % move closer to session 1.
- **Seed consistency** (median within-cell SD of MAE across seeds, session 1 → 2): new model D01 0.058 Nm vs
  0.5 s + retest map 0.111 Nm on CH, HM, IES, JM (10 seeds). Equal weighting alone ≈ 0.6× and auto-gain alone ≈ 0.7×
  the original model’s spread; window length does not matter (D02 ≈ D01).
- **Longer training check** (0.5 s + retest map, 6 people × seeds 0–2, 150-epoch cap / patience 25): spread 0.114 → 0.098 Nm,
  accuracy unchanged (MAE 0.88 Nm). The epoch cap is not the main reason D01 is steadier. Runs:
  `EMG_TQ_Modelling_runs/window50_retest_map_150ep`; the main 60 runs are in `EMG_TQ_Modelling_runs/window50_retest_map`.
