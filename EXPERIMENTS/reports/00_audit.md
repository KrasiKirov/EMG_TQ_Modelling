# Phase 0: audit of the project before any new experiment (2026-10-08)

## 1. How the original pipeline works
The code is in the **main checkout's working copy** (`ML/`). It is frozen in `EXPERIMENTS/baseline/original_code/`, with SHA-256 checksums in `original_code_sha256.txt`.

1. **Read** one `.flb` file per person. It contains MVC, passive and active trials for both sessions of one visit. Trials are sorted by their comment ("test" = session 1, "retest" = session 2). YES's labels are unreliable, so its trial numbers are hard-coded in `ML/config.py` (`TRIAL_OVERRIDES`).
2. **EMG envelope** (`ML/preprocessing/emg_envelope.py`):
   - band-pass 30–300 Hz, 4th order, run forwards and backwards (zero-phase);
   - absolute value;
   - 2 Hz low-pass.
3. **EMG scaling:** each channel is divided by its maximum over the session-1 trials.
4. **Target** = measured torque − passive torque. The passive torque comes from a table built from the passive trials of the same session:
   - each passive trial is cut at angle jumps;
   - a piece is kept if its torque spread is ≤ 20 Nm and its mean is ≤ ±20 Nm;
   - each kept piece is averaged, and nearby angles are merged;
   - the table is interpolated at each sample's angle.
   - YES uses lab-worksheet values (`PASSIVE_TORQUE_OVERRIDES`).
   - IES has no passive trials, so nothing is subtracted.
5. **Trim and downsample:** drop the first 1 s, then go from 1000 Hz to 100 Hz.
6. **Windows:** 20 time steps (0.2 s) × 5 features (4 EMG + angle). The target is the torque at the last step.
7. **Split:**
   - session 1: the first 80 % of each trial is used for fitting, the last 20 % for early stopping;
   - session 2: the whole session is the test set.

## 2. How LSTM-8 is trained and evaluated
- **Architecture:** LSTM(8) → Dropout 0.3 → LSTM(8) → Dropout 0.3 → Dense(1). That is 1,001 parameters.
- **Training:**
  - Nadam optimiser, learning rate 0.003, MSE loss in Nm², batch 8, at most 200 epochs;
  - ReduceLROnPlateau (factor 0.7, patience 10, minimum 1e-6);
  - EarlyStopping (patience 25, best weights restored).
- **No random seed is set.** Every run of the original code gives a different result.
- **Evaluation:** only one direction (learn session 1 → test session 2). Per position it reports R², VAF, RMSE, MAE and NRMSE, where NRMSE = RMSE ÷ (max − min). The scores are against the model's own target, which was built with the session-2 passive table.

## 3. Which positions and people perform poorly (prior evidence; reproduction with the original settings is running)
Old pipeline re-runs from this semester (faster training settings, 3 seeds, own target, S1 → S2). Problematic positions:

| Person | Position (rad) | R² |
|---|---|---|
| YES | −0.49 | −1.09 |
| JM | −0.64 | −0.04 |
| JM | −0.36 | 0.41 |
| JM | −0.08 | 0.45 (passive-table bug) |
| EG | −0.74 | 0.41 |

IES, CH and HM are ≥ 0.62 everywhere. The common factor: the 2–3 most plantarflexed positions, where the voluntary torque only varies by 0.4–0.8 Nm (std), against ~15 Nm at the strong positions.

## 4. Prior experiments (reused as evidence, not as proof)

| Where | What | Caveat |
|---|---|---|
| `notes/report/` (report 1) | ~40 variants. The passive-table bug was found, plus EMG size change between sessions, loss imbalance, and seed instability. Angle-sensor leak test: no leak. Other network types: none better. | Changes often bundled; a mix of "sandbox" and real code. |
| `ML/research/` E01–E35 (report 2) | Single and combined changes, 3 seeds, both directions, 6 people. | **Faster training settings** (batch 64, not 8). Emphasis on an "in-trial" answer key the supervisor did not adopt. Final choice made after seeing the extra people. |
| `ML/research/` E36–E38 | Passive-trials-only versions under the supervisor's rule. | Faster settings; 1 configuration per idea. |
| `notes/final_pipeline/results_tables/all_results.xlsx` | Every prior run re-scored against three answer keys. | — |

What prior work established, to be re-tested here with the original settings and one change at a time:
- loss weighting per position and EMG auto-gain gave the largest gains;
- dropout, learning rate and model size were never varied on their own.

## 5. Problems found
1. **The committed code is not the code that was used.**
   - git HEAD has 2 people, no YES overrides, and a 10 Nm passive bound.
   - The working copy (never committed) has 6 people, the YES overrides and a 20 Nm bound.
   - The baseline here is the working copy, frozen. **Please confirm** this is what you ran last semester.
2. **Passive-table bug (JM):** the 20 Nm rule accepted pieces of passive trials that were not relaxed, giving a +11.7 Nm entry near −0.07 rad.
3. **Small mismatch between the experiment framework and the original:** the framework's validation part has 1 extra window per trial. The new runner removes it. Otherwise the data was shown to be identical (`code/check_equivalence.py`: fitting and test data match to the last digit).
4. **No seeds in the original**, so single results can't be reproduced. Fixed seeds (0, 1, 2) are used here. That changes nothing in the method.
5. **The two NRMSE definitions differ:** the original uses RMSE ÷ (max − min), report 2 used the 98–2 percentile range. EXPERIMENTS uses the original definition.
6. **Grading against a model's own target hides passive-table errors.** Every model is therefore also graded against one fixed answer key built from the passive trials.
7. **IES has no passive trials,** so its "voluntary" target is the raw measured torque. This is flagged in every table.
8. **CH pushes in both directions,** and at one position the push direction flips between sessions.
9. **The sessions are in the same visit with the electrodes left on.** This is the easiest form of session transfer.
10. **The zero-phase filters use about 0.25 s of future signal,** which makes the pipeline offline-only. This is the same for every experiment.

## 6. Simplest explanations for the poor plantarflexed performance (to test)
1. **Low target variance.** The same error in Nm gives a much lower R² there, so we also look at RMSE and the reference predictor.
2. **Passive-torque mismatch.** The passive-trial value differs from the relaxed level inside the trial by 0.6–2 Nm at those positions; the table is too high in 20 of 20 trials. Under the supervisor's key this offset is part of the target.
3. **EMG size changes between sessions** for the same push (JM ≈ 0.4–0.5× at some positions).
4. **Loss imbalance.** Plain least squares in Nm² lets the strong positions dominate (YES: 1 % of the loss at its 2 most plantarflexed positions).
5. **Training variability.** Runs disagree, especially for JM.

## 7. First experiments, simplest first (one change each from LSTM-8)

| ID | Change | Tests explanation |
|---|---|---|
| X01 | clean passive table (static holds) | 2 (bug part) |
| X02a/b/c | dropout 0.0 / 0.1 / 0.5 | over/under-fitting (prior evidence weak, cheap check) |
| X03 | learning rate 0.001 | 5 |
| X04 | every position counts the same (divide by typical swing) | 4 |
| X05 | EMG auto-gain | 3 |
| X06 | 0.4 s window | context length |

Tier 2, only if justified: 16 units, or position-conditioned corrections.

## 8. Hardware
- **GPU:** an RTX 3070 is present (driver 610.88), but **TensorFlow 2.21 on native Windows has no GPU support** (TF ≥ 2.11). PyTorch is not installed.
- **Why CPU is fine here:** LSTM-8 has 1,001 parameters and is trained with batch 8, so each step is tiny and GPU launch overhead would dominate. We run 14 single-thread CPU jobs in parallel instead (16 cores).
- **To benchmark the GPU** we would need PyTorch with CUDA (~2.5 GB download, plus validating a re-implementation) or TensorFlow in WSL2. **Your call**; not needed for correctness.
- **Versions:** Python 3.11.1, TensorFlow 2.21.0, NumPy and SciPy as in `.venv311`.

## Decisions made (reversible; please confirm)

| Decision | Choice | Why |
|---|---|---|
| Baseline code | main working copy, frozen | what was actually run last semester |
| Primary answer key | passive trials of the tested session, clean static holds | supervisor's rule, plus the JM bug fixed in the key |
| Also reported | the original target (with the bug) | fidelity to last semester |
| Direction | S1 → S2 primary; S2 → S1 supplementary | original protocol is S1 → S2 |
| Seeds | 0, 1, 2 | the original had none |
