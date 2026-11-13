# Full-trial inspection plots: old pipeline vs FINAL

Copies of the full-trial inspection plots, renamed so you can tell what each one is. The originals are under `ML/research/results/` (see the bottom of this page).

## File names

`<SUBJECT>_learned<S1|S2>_tested<S2|S1>_<OLD|FINAL>.png`

| part | meaning |
|---|---|
| `YES`, `JM`, `EG` | the 3 main subjects (used to compare and choose) |
| `IES`, `CH`, `HM` | the 3 held-out subjects (checked afterwards) |
| `learnedS1_testedS2` | trained on session 1, then predicting session 2, which it never saw |
| `learnedS2_testedS1` | the other way round |
| `OLD` | the original pipeline (config E01_old) |
| `FINAL` | the recommended pipeline |

The FINAL plots come from running `--config FINAL` on every subject (`ML/research/results/final_check/`). In all 12 runs the predictions are identical to the earlier E22 runs (E32 for CH, who pushes both ways, so auto-gain is switched off).

All plots are seed 0 and scored against the **shared reference truth**: measured torque minus the relaxed level found inside the same trial.

## How to read one plot

- **8 panels = 8 trials = 8 ankle angles** of the tested session. The **top panel is the most toes-down (plantarflexed) angle**, which is the hard one. Going down, the angles move towards toes-up.
- **Panel title:** `Position −0.486 | R²=… | RMSE=… Nm | N=…`
  - position = ankle angle in rad
  - R² = 1 is perfect, 0 is no better than guessing the average, below 0 is worse
  - RMSE = the typical error in Nm
  - N = number of 200 ms windows scored
- **Blue line = the truth** (voluntary torque). **Red line = the model's prediction.**
- **x-axis** = time in the ~90 s trial. **y-axis** = active torque in Nm. Negative = pushing the toes down. ~0 = relaxed.

## What to look for

| what you see | what it means |
|---|---|
| red follows blue, and both sit at ~0 between pushes | good |
| red has the right shape but is shifted up or down the whole time | an **offset** error. Typical of the OLD pipeline: a wrong passive estimate |
| red goes too deep or not deep enough on the pushes | a **gain** error: the EMG size changed between sessions |
| red wiggles when blue is flat at 0 | **phantom torque**: the model "sees" force while the person is relaxed |
| a panel has negative R² but a small RMSE | the person barely pushes at that angle (~1 Nm), so even a small error gives a bad R². Check the RMSE in Nm |

**Good pairs to show:**
- `YES_learnedS1_testedS2_OLD` vs `…_FINAL`: in the top panel, the old one is ~1.5 Nm too low; FINAL follows the pushes.
- `JM_learnedS2_testedS1_OLD` vs `…_FINAL`: the same effect at JM's toes-down angles.
- `IES_learnedS2_testedS1_OLD` vs `…_FINAL`: IES has no passive recordings, and FINAL still works.

## Where the originals are

`ML/research/results/<study>/<subject>/<S1toS2|S2toS1>/<config>/`, where `<study>` = `screen` (YES/JM/EG) or `confirm` (IES/CH/HM).

Each config folder has two kinds of plot:
- `full_trial_inspection_truth_<tested session>_fast_seed0.png`: against the shared truth (these are the ones copied here);
- `full_trial_inspection_<tested session>_fast_seed0.png`: against that config's own target, the old style.

In the file names, `fast` = the training settings used (batch 64) and `seed0` = the random seed. Only seed 0 has plots.
