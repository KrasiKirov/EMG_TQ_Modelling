# Improving LSTM-8 at the toes-down positions: final report (2026-10-09, revised after an independent fact-check)

Plain-English summary of the controlled experiments in `EXPERIMENTS/`. Model metrics come from `results/all_evaluations.csv` (one row per experiment, subject, direction, seed, position and answer key). Diagnostics come from `results/diagnostics_X00.csv` and `results/offset_shift.txt`.

**Two baselines appear in this report; they are not interchangeable:**
- **X00:** the frozen original code. It is the official baseline and only exists for learned S1 → tested S2.
- **X00f:** the same baseline run through the experiment framework. It uses identical data, but training is not bit-identical (§1). Experiments were compared with X00f, using the same code path and the same seeds.

**How R² is aggregated:**
- Experiment-versus-baseline tables use **matched seeds** and **unfloored** R², as the pre-written criteria do.
- Summary tables that average positions floor each position's R² at −1 and say so.

---

## 1. Problem and baseline
- **Task:** LSTM-8 predicts voluntary ankle torque from 4 EMG signals plus the ankle angle.
  - It learns from one session and predicts the other.
  - There are 8 fixed ankle angles ("positions") and 6 people.
- **Last semester's finding:** R² below 0 at the 2 most toes-down (plantarflexed) positions of some people.
- **Baseline X00:** the original code, frozen from the project folder on disk (`baseline/original_code/`), with 3 fixed seeds. The original code set no seed.
  - For the 2 values we can compare (seed 0), it matches the earlier runs of the original code with the original settings: YES −0.49 rad R² −0.68, and JM −0.64 rad −0.32.
- **X00f vs X00:**
  - The framework builds the same data (`code/check_equivalence.py`). It differs only by rounding at the 1e-6 Nm level, plus one validation window per trial, which the runner removes.
  - Training is chaotic, so results differ at the level of run-to-run variation. Example: X00f JM S1 → S2 toes-down R² is 0.09, against 0.39 for X00.
- **Answer key (supervisor's choice):** measured torque − passive torque from the **passive trials of the session being tested**.
  - We use the clean, still holds of the passive trials, which removes JM's bad table entry.
  - For YES, the lab worksheet values are used, as in the original code.
  - IES has no passive trials, so nothing is subtracted.
- **Metrics,** for every subject, position, direction and seed:
  - R²;
  - MSE (Nm²);
  - RMSE (Nm);
  - NRMSE = RMSE ÷ (max − min), the original definition.

## 2. Why the toes-down positions fail (`02_diagnostics_tables.md`, `results/offset_shift.txt`)
1. **The predictions are not random.** Across the 12 toes-down positions they correlate with the key at 0.80–0.99, and the best time alignment is within −2 to +3 samples (≤ 30 ms).
2. **A wrong level (offset) is often the largest part of the error.**
   - It is 54–72 % of the squared error at 8 of the 12 toes-down positions: YES −0.49, EG ×2, IES ×2, HM ×2 and JM −0.64.
   - It is 18–40 % at the other 4: YES −0.38, JM −0.79 and CH ×2.
3. **At several toes-down positions the target barely moves.** The standard deviation is 0.40 Nm at YES −0.49, 0.60–0.78 Nm for JM and 0.77 Nm at HM −0.79, against up to ~28 Nm at strong positions. Elsewhere it is 1.9–8.2 Nm. A small error in Nm therefore gives a very negative R² at the low-variance positions.
4. **The EMG can come out a different size in the second session for the same push.** JM ≈ 0.4–0.5× at some positions; from an earlier analysis, `00_audit.md`.
5. **Plain least squares lets the strong positions dominate the fit.**
6. **Results swing a lot between training runs.** YES −0.49 gave R² −0.68 / −0.26 / −1.80 with seeds 0 / 1 / 2.
7. **One labelling bug:** JM's original passive table had a +11.7 Nm entry made from pieces of passive trials that were not relaxed (`00_audit.md`).
8. **YES −0.49 (learned S1 → tested S2): the resting level of the key shifts by 0.72 Nm between sessions,** while the key's standard deviation is only 0.40 Nm.
   - A model that tracked every push perfectly but kept session 1's resting level would score R² −2.3.
   - Real runs land between −1.4 and +0.5. The result there depends largely on a level shift the EMG cannot reveal.
   - **This is not an upper bound** (an earlier version wrongly called it a "hard ceiling").

## 3. Experiments, in order, and what the pre-written rules said
The rules were written before any result (`SUCCESS_CRITERIA.md`). Each one compares an experiment with its reference over matched seeds.
1. Problematic R² rises by ≥ 0.10 and problematic RMSE does not rise.
2. R² improves in ≥ 3 of 4 problematic cells.
3. No other position loses > 0.05 R², the mean other-position R² drops ≤ 0.01, and no other position's RMSE rises > 15 %.
4. The gain is > 2 × the reference's run-to-run SD at the problematic positions.

Stage B also requires the same sign in the reverse direction. Stage C requires each added change to pass rules 1–4 against the previous configuration.

### Stage A: one change at a time on the 2 worst people (YES, EG), learned S1 → tested S2, seeds 0–1
Baseline problematic R² = 0.344.

| ID | change | reason | problematic R² | Δ RMSE | rules 1/2/3/4 |
|---|---|---|---|---|---|
| X02a/b/c | dropout 0 / 0.1 / 0.5 (instead of 0.3) | over/under-fitting check (Gal & Ghahramani 2016; Son et al. 2024) | 0.17 / 0.16 / 0.17 | worse | ✘ |
| X03 | learning rate 0.001 (instead of 0.003) | runs were unstable | 0.335 | +0.01 Nm | ✘ |
| X04 | every position counts the same in the fit (each position's target ÷ its standard deviation = weighted least squares) | loss imbalance (Kendall et al. 2018; Yang et al. 2021) | 0.431 (+0.09) | −0.50 Nm | ✘ (rule 1) |
| X05 | EMG auto-gain (each muscle rescaled by its own recent quiet and strong levels, using only past EMG; off for people who push both ways) | EMG size change between sessions (Besomi et al. 2020; Tanaka et al. 2022) | 0.527 (+0.18) | −0.56 Nm | ✔ ✔ ✔ ✘ (noise) |
| X06 | 0.4 s window (instead of 0.2 s) | more history (Wang et al. 2024) | −0.21 | worse | ✘ |

- **Strictly, no Stage A change qualified for Stage B.**
- **We continued with X05, X04 and X03 anyway, as a judgment call.** X05 failed only the noise rule, which is hard to pass with 2 seeds when one position (YES −0.49) swings from −1.4 to +0.1. X04 and X03 had large RMSE drops (X04) or looked promising before the seed mismatch was corrected (X03).
- **Dropout and window:** in these runs no improvement was detected. The changes are within run-to-run noise, so this does not prove the baseline is "correctly" regularised.
- **X01, clean passive table (JM only; it changes nothing for the others):**
  - At JM −0.08 it fixes the labelling bug: R² −54.9 → 0.58, error 12.5 → 1.1 Nm.
  - The rules are written for the toes-down positions, and there it is "not promising" (the toes-down cells moved within noise).
  - Its JM +0.055 position got worse (−0.09 R², +16 % RMSE).
  - It is kept because it corrects a labelling error. That is a judgment, not a rule outcome.

### Stage B: 3 seeds (0–2) + the reverse direction
Baseline problematic R² = 0.156.

| ID | problematic R² | Δ RMSE | worst other-position RMSE | reverse-direction Δ | rules 1/2/3/4 | reverse sign |
|---|---|---|---|---|---|---|
| X05 auto-gain | 0.297 | −0.63 Nm | +9 % | −0.06 | ✔ ✔ ✔ ✘ | ✘ |
| X04 equal weight | 0.420 | −0.67 Nm | +36 % | +0.02 | ✔ ✔ ✘ ✘ | ✔ |
| X03 learning rate | 0.425 | −0.19 Nm | +15.004 % | +0.10 | ✔ ✔ ✘ ✘ | ✔ |
| C01 = X05 + X04 | 0.407 | **−0.77 Nm** | +4 % | −0.05 | ✔ ✔ ✔ ✘ | ✘ |

**No experiment passed Stage B.**

**Stage C step rule (C01 vs the previous configuration, 3 seeds):**
- vs X05: +0.11, 4 of 4 cells, fails the noise rule; reverse direction +0.01 with +35 % RMSE somewhere.
- vs X04: −0.01.

So C01 also does not pass the Stage C rule, and combining X04, a "trade-off", was not discussed beforehand as the rules require. **The final candidate was therefore chosen by judgment, not by the pre-written path.** The reasons:
- the lowest problematic RMSE;
- no position with a large regression;
- consistent gains at EG −0.74, EG −0.60 and YES −0.38.

Per position, learned S1 → tested S2, R² per seed:

| position | X00f baseline | C01 |
|---|---|---|
| EG −0.74 | 0.01 / −0.18 / −0.79 | 0.96 / 0.96 / 0.97 |
| EG −0.60 | 0.84 / 0.89 / 0.79 | 0.97 / 0.97 / 0.96 |
| YES −0.38 | 0.73 / 0.86 / 0.49 | 0.86 / 0.91 / 0.89 |
| YES −0.49 | −0.52 / 0.12 / −1.36 | −1.36 / −1.20 / −1.00 |

In the reverse direction C01 is also better at EG −0.74 (0.93–0.95) and EG −0.60. YES −0.38 improves on average, but seed 2 goes from 0.96 to 0.93. YES −0.49 gets worse (baseline 0.40 / 0.57 / 0.15, C01 −0.69 / −0.29 / −0.39). A likely reason is that YES barely activates the calf there, so auto-gain also amplifies noise.

### Tier 2: aimed at the YES −0.49 offset
**T01:** the passive-trial value of each trial as an extra input. It is measured before the trial, so it is available at prediction time.
- **Seeds 0–1, YES + EG:** problematic R² 0.344 → 0.53.
- **YES −0.49 improves** (learned S1 → tested S2: −0.52 / 0.12 → −0.00 / 0.39). X03 also improved this position, from −0.59 to −0.10 with 3 seeds.
- **Other positions get worse,** up to +197 % RMSE at YES +0.245 in the reverse direction (R² 0.969 → 0.693). Passive values also change between sessions at strong positions (YES +0.25: −10.1 → −5.0 Nm), and the model turns that into wrong level shifts.
- **Rejected.**

### Stage D: final candidate on the 4 people never used for choosing (JM, IES, CH, HM)
**D01 = clean passive table + auto-gain + equal weight per position.** For YES and EG it is identical to C01. It was run in both directions with 3 seeds.

| comparison | problematic R² | cells improved | Δ RMSE | worst other-position RMSE | rules 1/2/3/4 |
|---|---|---|---|---|---|
| vs X00f, S1 → S2 | 0.632 → 0.926 | 8/8 | −0.46 Nm | +20 % | ✔ ✔ ✘ ✘ |
| vs X00f, S2 → S1 | 0.746 → 0.929 | 6/8 | −0.27 Nm | +32 % | ✔ ✔ ✘ ✔ |
| vs official X00, S1 → S2 | 0.706 → 0.926 | 8/8 | −0.43 Nm | +18 % | ✔ ✔ ✘ ✘ |

## 4. Results: official baseline (X00) vs the final candidate, learned S1 → tested S2
Toes-down = mean of the 2 most toes-down positions; seed-mean; R² floored at −1 per position.

| person | R² X00 → final | RMSE X00 → final (Nm) |
|---|---|---|
| YES | −0.08 → −0.06 (unfloored −0.08 → −0.15) | 0.82 → 0.67 |
| EG | 0.25 → **0.96** | 2.08 → 0.55 |
| JM | 0.39 → **0.84** | 0.53 → 0.27 |
| IES | 0.64 → **0.96** | 1.75 → 0.63 |
| CH | 0.96 → **0.98** | 1.09 → 0.86 |
| HM | 0.83 → **0.92** | 0.49 → 0.37 |
| **All 6, toes-down** | **0.50 → 0.77** | **1.13 → 0.56** |
| **All 6, other positions** | **0.87 → 0.97** | **1.83 → 1.19** |

The reverse direction (vs X00f, `results/final_summary_tables.md`) gives toes-down 0.71 → 0.82 and other positions 0.84 → 0.95. Fewer positions improve in this direction: 9 of 12 toes-down and 22 of 36 others.

**Every regression vs X00f** (seed-mean R² drop > 0.01 or RMSE rise > 10 %; `results/final_corrections_check.txt`):

| direction | position | R² | RMSE |
|---|---|---|---|
| S1 → S2 | **YES −0.49 (toes-down)** | −0.59 → −1.19 | +19 % |
| S1 → S2 | HM −0.22 | 0.980 → 0.972 | +20 % |
| S1 → S2 | CH +0.24 | 0.990 → 0.988 | +11 % |
| S1 → S2 | HM +0.05 | 0.985 → 0.981 | +10 % |
| S2 → S1 | **YES −0.49 (toes-down)** | 0.38 → −0.46 | +54 % |
| S2 → S1 | HM +0.05 | 0.965 → 0.938 | +32 % |
| S2 → S1 | YES −0.07 | 0.992 → 0.986 | +30 % |
| S2 → S1 | CH −0.05 | 0.991 → 0.988 | +20 % |
| S2 → S1 | YES +0.03 | 0.983 → 0.977 | +18 % |
| S2 → S1 | JM −0.50 | 0.872 → 0.836 | +13 % |
| S2 → S1 | YES +0.25 | 0.969 → 0.961 | +13 % |
| S2 → S1 | EG +0.10 | 0.984 → 0.980 | +11 % |
| S2 → S1 | IES +0.14 | 0.979 → 0.975 | +10 % |

In S2 → S1, HM's toes-down positions dip slightly: 0.917 → 0.910 and 0.948 → 0.946.

**Bottom line, against our own rules:**
- **Formally, no change is confirmed.** The noise and regression rules fail at several stages, and the final combination was chosen by judgment.
- **What the data does show:**
  - At the toes-down positions of 5 of 6 people, the final candidate brings large gains on the 4 people never used for choosing (S1 → S2: 0.71 → 0.93, 8 of 8 positions better).
  - It roughly halves the error there.
  - It slightly improves most other positions.
  - It costs small RMSE increases (≤ 32 %, R² changes ≤ 0.035) at some strong positions.
  - **YES −0.49 is worse.**

## 5. Literature (full list: `literature/references.md`, 30 verified papers)
- **Literature only motivated the experiments; our own runs decided.**
- **Comparable (EMG/MMG → torque):**
  - Son et al. 2024: dropout sweep, isometric ankle, MMG.
  - Truong et al. 2023: an LSTM with the same 2 × 8 units, dropout 0.3, batch 8 and Nadam 0.003, estimating lower-limb moments.
  - Zhang et al. 2023: LSTM vs neuromusculoskeletal models.
  - Liu, Liu & Clancy 2015: joint angle changes the EMG–torque relation at the elbow.
  - Schulte et al. 2022: models degrade across days.
  - Kizyte et al. 2023: ankle torque from HD-EMG vs bipolar EMG. It does not report per-angle accuracy, so it is context only.
- **Related techniques, from other problems:**
  - Kendall et al. 2018 and Yang et al. 2021 (loss weighting);
  - Tanaka et al. 2022 (sliding-window EMG normalisation);
  - Gal & Ghahramani 2016 (dropout in RNNs);
  - Bouthillier et al. 2021 (run-to-run variance).
- **None of them tested this exact setting:** 8 isometric ankle angles, the second session of the same visit, and a passive-trial answer key.

## 6. Limitations
- **Only 6 people.** The two sessions are in the same visit with the electrodes left on, which is easier than a different day.
- **The pre-written decision path was not followed to the end** (see §3). The validation on 4 unused people is the strongest evidence. The final choice was still made by judgment.
- **YES −0.49 is dominated by a resting-level shift** the EMG cannot reveal, and the candidate makes it worse.
- **IES has no passive trials,** so its target is the raw measured torque.
- **Few seeds:** 2 in screening, 3 in confirmation. Run-to-run variance is large at the toes-down positions, and training is chaotic (1e-6 differences change the outcome).
- **Zero-phase EMG filters,** so this is offline only. That is the same for every model compared.
- **The original slow training settings were used everywhere:** 156 runs, 25–220 min each, about 224 CPU-hours in total.

## 7. Next steps
1. **5 or more seeds** for the final candidate vs X00, to settle the noise rule.
2. **Auto-gain only for muscles clearly active at a position,** to avoid amplifying noise as at YES −0.49 (a new single-change experiment).
3. **A session recorded on a different day.**
4. **A short relaxed hold right after each pushing trial,** to measure the resting-level shift.

## 8. Reproduce
```bash
source ML/experiments/env.sh
$PY EXPERIMENTS/code/run_queue.py --original-baseline --subjects YES JM EG IES CH HM --seeds 0 1 2
$PY EXPERIMENTS/code/run_queue.py --exps X00f_baseline_framework D01_cleanpassive+autogain+equalweight \
    --subjects YES JM EG IES CH HM --directions S1toS2 S2toS1 --seeds 0 1 2
$PY EXPERIMENTS/code/score.py && $PY EXPERIMENTS/code/diagnostics.py && $PY EXPERIMENTS/code/offset_shift.py
$PY EXPERIMENTS/code/analyze_stage.py --stage D --seeds 0 1 2 --subjects JM IES CH HM --exps D01_cleanpassive+autogain+equalweight
$PY EXPERIMENTS/code/final_figures.py
```

**Where everything is:**
- **Figures:** `plots/final_*.png`.
- **Per-stage reports:** `reports/stage*_*.md`, regenerated with matched seeds.
- **Ledger:** `experiment_ledger.csv`.
