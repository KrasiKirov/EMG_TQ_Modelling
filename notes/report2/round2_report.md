# Round 2: plantarflexed angles, a fair scoring rule, and 23 experiments

*Follow-up to [report 1](../report/dorsiflexion_report.md). What I changed in how I score the model, what I tried (with sources), and what actually helps the plantarflexed angles when the model is tested on a session it never saw.*

- **Date:** 2026-10-05
- **Branch / worktree:** `claude/dorsiflexion-r2-drop-341137` (still nothing committed)
- **Subjects:** YES and JM (the bad ones) + EG (a third subject I never used to design anything). IES, CH and HM were held out from every decision of this round (section 9). They aren't completely fresh data: all three were analysed in report 1, CH's push-direction flip shaped report 1's design, and their recordings were pre-training data for the two transfer-learning configurations (E20, E21).
- **New code:** everything is in a new folder, `ML/research/`. It imports the old pipeline and doesn't change a single line of it (section 12).

**Contents:** [TL;DR](#tldr) · [1. What you asked](#1-what-you-asked-for-this-round) · [2. Setup recap](#2-setup-recap) · [3. Fair scoring](#3-the-first-problem-the-old-scoring-was-unfair) · [4. Relaxed periods](#4-your-question-are-the-relaxed-neutral-parts-just-noise) · [5. What I tested](#5-what-i-tested-and-why) · [6. How I judged it](#6-how-i-judged-it-without-cheating) · [7. Results](#7-results) · [8. The plantarflexed angles up close](#8-the-plantarflexed-angles-up-close) · [9. Held-out subjects](#9-check-on-3-subjects-held-out-from-this-rounds-choices) · [10. What I'd use](#10-what-id-use) · [11. Limitations](#11-limitations) · [12. Code](#12-the-code) · [13. Sources](#13-sources) · [Appendix](#appendix)

---

## TL;DR

- **The old scoring was unfair, and it was hiding failures.** Each pipeline used to be graded against its *own* target (measured torque minus its *own* passive estimate), so a pipeline with a wrong passive estimate was graded against a wrong answer key. I built one **common reference truth** for everyone: measured torque minus the relaxed ("passive") level measured *inside the same trial*, during the moments the EMG says the person is relaxed (section 3).
  - Under that truth, the old pipeline drops from R² 0.68 to **0.54**.
  - At JM −0.078 rad its prediction was off by **~12–13 Nm** (R² −60), and the old score didn't show it.
- **Your relaxed periods are useful, not noise.** Removing them from training doesn't change R², but the model then "hallucinates" torque while the person is relaxed (phantom torque at rest 0.25 → **0.36 Nm**) (section 4).
- **The plantarflexed angles are much better on YES / JM / EG, but YES −0.486 tested on session 2 is still weak.**
  - I ran 23 configurations, all with 3 subjects × 2 train/test directions × 3 seeds (414 runs).
  - Every serious candidate is far ahead of the old pipeline, with **no negative per-angle R² anywhere** (the old one had 8 / 48).
  - On the 2 most plantarflexed angles: **+0.80 R²** on average vs the old pipeline (95% CI +0.47 to +1.15), better in **12 / 12** cases.
- **Your angles** (R², mean of 3 seeds, old → **E22, the pipeline I recommend**, with [E03, report 1's final] in brackets):
  - YES −0.486, trained on session 1 → tested on session 2: **−14.3 → 0.34** [0.47] (error 1.52 → 0.31 [0.28] Nm). Still the weakest cell.
  - YES −0.486, the other direction: **−7.5 → 0.70** [0.72]
  - JM −0.787: **0.18 → 0.91** [0.90] and **−0.99 → 0.90** [0.76]
  - JM −0.644: **0.50 → 0.91** [0.88] and **−0.44 → 0.79** [0.78]
  - The predictions now follow the pushes instead of the "gibberish" you saw (section 8).
- **A selection rule picked E29** (E03 with a 400 ms window instead of 200 ms). I wrote the rule down after most screening results were in, but before opening the held-out results. E29 scores 0.932 vs 0.925 for E03, which is *not* significant after correcting for the number of comparisons. The longer window helps the strong angles a bit, not the plantarflexed ones.
- **The check on 3 held-out subjects (IES, CH, HM) changed my recommendation.**
  - **IES has no passive recordings at all.** E03 and E29 then collapse at IES's plantarflexed angles (R² ~0.05–0.08 on average); the old pipeline fails there in one direction only. The **rest-anchored labels (E22)** don't need any passive recording: IES R² **0.97–0.99 at every angle**.
  - Like for like (all 5 pipelines on the same test trials, only CH's flipped-task angle left out): E22 **0.967**, E31 0.951, E29 0.880, old 0.878, E03 0.870.
- **So I'd use E22**: report-1 final pipeline + rest-anchored labels, with report 1's "auto" rule that switches auto-gain off for subjects who push in both directions (only CH).
  - On YES / JM / EG it ties E03 and E29 on the plantarflexed angles (0.84), with less phantom torque at rest.
  - **Caveats:**
    - this choice was informed by the held-out check, so that check is no longer independent evidence for E22;
    - E22's training labels are built exactly like the scoring truth, which gives it a built-in advantage on phantom torque;
    - the IES mechanism rests on one subject (sections 9–11).
- **What did NOT help:**
  - your holds (hold level, relaxation drift)
  - MVC normalisation
  - extra plantarflexed weighting (×3 tie, ×10 worse)
  - EMG gain augmentation on top of auto-gain
  - pre-training on the other 5 subjects (worse)
  - ridge regression (worse)
  - per-session EMG scaling (broke JM)
  - removing or down-weighting the relaxed windows
- **Most of the fix is data processing, not model size.** The original tiny LSTM + EMG auto-gain + rest-anchored labels (E31) is only ~0.01 R² behind on YES / JM / EG. It is clearly weaker on held-out HM (0.895 vs 0.930) and sometimes overshoots.
- **CH pushed toes-*up* at one angle in session 1 and toes-*down* at the same angle in session 2.** The auto-gain breaks there. With auto-gain off (what report 1's "auto" rule does for CH), E22 gets CH right: 0.99 instead of 0.74 (section 9).

---

## 1. What you asked for this round

| Your request | Where |
|---|---|
| Are the relaxed / neutral parts of the active trials noise or useful? | Section 4 |
| Use the passive holds before each trial and the MVC trials | Sections 5 and 7 (E04, E06, E10, E11) |
| YES −0.486 looks like gibberish, fix it | Section 8 |
| Is adding the retest session good for a more general model? | Not tested: you said to skip it if the results are satisfactory. My earlier answer in chat (good for a final model, but measure it with cross-session tests) is still untested (section 11) |
| Search the literature, cite what works | Sections 5 and 13 |
| Best practices, no invented results, document the code, append don't modify | Sections 6 and 12 |
| Save the trial inspection plots | Section 8 (the seed-0 run of every configuration, subject and direction has them) |
| YES, JM + another person | YES, JM, EG (+ IES, CH, HM as a held-out check) |

---

## 2. Setup recap

Same experiment as report 1 (see [report 1, section 1](../report/dorsiflexion_report.md#1-the-setup)):

- 4 EMG envelopes (GM, GL, SOL, TA) + ankle angle → **active (voluntary) ankle torque**
- 200 ms windows at 100 Hz, one model per subject

What's new this round:

- **Both directions.** Every configuration is trained on session 1 and tested on session 2 (`S1toS2`), *and* trained on session 2 and tested on session 1 (`S2toS1`). The two sessions are the lab's "test" and "retest" blocks, recorded one after the other in the same visit (same file), not on different days. That's a 2-fold cross-session test; report 1 only did S1→S2.
- **3 seeds for every screening run (YES / JM / EG).** Seed = the random starting point of training. Results are the **mean of the 3 seeds**; I never pick the best seed. The held-out check (IES, CH, HM) used seed 0 only.
- **Training speed setting.** The faster batch-64 "screening" setting from report 1. Report 1 showed the old-vs-new conclusions also hold with your original batch-8 setting.
- **Angle names.** Negative angle = **plantarflexion** (toes down), as found in report 1. The "weak" angles are the 2 most plantarflexed angles of each subject.

---

## 3. The first problem: the old scoring was unfair

### What was wrong

The target is "measured torque − passive torque". Every pipeline estimates the passive part its own way, so every pipeline also had **its own test target**. That isn't a fair comparison:

- If a pipeline's passive estimate is wrong, its test target is wrong in the same way, and its own score can't see the problem.
- Two pipelines with different passive estimates are graded against different answer keys.

### The fix: one reference truth for everyone

During an active trial there are moments where the person is relaxed: not pushing, EMG near zero. In those moments the measured torque **is** the passive torque. So:

1. Find the relaxed moments from the **EMG only**: when the average activity of the 4 muscles (each scaled to its own range) stays near zero. This is the same EMG-only rest detector as in report 1; it never uses torque.
2. Draw a smooth curve B(t) through the torque during those moments (8 s smoothing). That's the passive level *inside this trial*. It follows the slow relaxation drift and the session-specific level.
3. **Reference truth = measured torque − B(t).** Every configuration's predictions are scored against this same truth.

The model **never sees** this truth as an input; it only predicts from EMG + angle. The truth just defines what the right answer is, in the most physical way available. It does use the measured torque of the test trial, but that's fine: the measured torque *is* the thing we're predicting.

![Reference truth for JM: measured torque, EMG-detected rest, in-trial baseline, old passive levels](figures/truth_inspection_JM.png)

*Figure 1: JM, every trial of both sessions. Grey = measured torque, green dots = moments the EMG says "relaxed", black = the in-trial passive baseline B(t). Dashed blue = static passive map, dotted yellow = mean of the hold before the trial. At the plantarflexed angles (top rows) the relaxed torque sits **below** the map and hold values. Same figure for the other 5 subjects: `figures/truth_inspection_<subject>.png`.*

### What it revealed

![Own target vs reference truth for the old pipeline](figures/fig02_own_target_vs_truth.png)

*Figure 2: the old pipeline (E01) on three trials. Blue = its own test target, black = reference truth, orange = its prediction. The R² values in the legend are seed 0.*
- *Top, YES −0.486: the own target is shifted ~2 Nm down, and so is the prediction. It looks "OK-ish" against its own target (R² −1.1) but is very wrong against the truth (R² −14.5).*
- *Middle, JM −0.078: the passive-map bug from report 1. Target and prediction are both ~12–13 Nm off, so the old score (R² +0.54) completely hides it.*
- *Bottom, JM −0.787: the prediction is ~0.9 Nm too low on average.*

| Old pipeline (E01), mean over 3 subjects × 2 directions × 3 seeds | own target | reference truth |
|---|---|---|
| mean per-angle R² (each angle floored at −1) | 0.683 | **0.540** |

The new pipelines score about the same under both (E03: 0.917 own vs 0.925 truth), because their passive estimate was already close to the truth. **From here on, every number is against the reference truth.** The own-target numbers are in the appendix for comparison with report 1.

> **Honesty notes:**
> - I decided on this new scoring after the literature search, having seen only the old pipeline's results. All configurations are scored with it, including the old one.
> - **Circularity caveat:** the rest-anchored configurations (E22, E23, E31) build their *training* labels with exactly this construction (same EMG rest detector, same 8 s baseline). They are trained toward the scoring definition itself, and phantom torque is measured on the same rest moments where their labels are zero by construction. Part of their advantage on phantom torque is therefore built in. On YES / JM / EG their R² is no better than E03's, so it doesn't inflate the main comparison, but a truth built with a different rest detector or smoothing would be a good extra check.

---

## 4. Your question: are the relaxed ("neutral") parts just noise?

**Short answer: no, they're useful. Keep them.**

- The target is *voluntary* torque, so during relaxed moments the right answer is **0 Nm**, and the model has to learn that. Those windows are exactly where it learns "quiet EMG → zero torque".
- What *was* noisy in the old pipeline was the **label** in those moments, not the EMG. The passive level it subtracted (the hold value) was too high for YES, so "relaxed" was labelled as −1 to −2 Nm instead of 0. That's a passive-estimate error, not a problem with the relaxed periods.

**The test.** I trained E03 three ways: all windows, relaxed windows weighted ×0.2 (E25), and relaxed windows removed (E24). I also tried the rest-anchored labels (E22), where the relaxed moments are labelled 0 by construction.

![Relaxed-window ablation](figures/fig04_rest_ablation.png)

*Figure 3: left = R² on all windows, middle = R² on the active (pushing) windows only, right = phantom torque, the average size of the prediction while the person is relaxed (should be 0). Dots = subject × direction.*

| | R² all | R² active only | phantom torque at rest |
|---|---|---|---|
| E03, all windows | 0.925 | 0.90 | 0.25 Nm |
| E25, relaxed ×0.2 | 0.923 | 0.90 | 0.29 Nm |
| E24, relaxed removed | 0.924 | 0.90 | **0.36 Nm** |
| E22, relaxed labelled 0 (rest-anchored) | 0.925 | 0.89 | **0.17 Nm** |

Removing the relaxed windows doesn't make the pushing part any better (same R² on active windows), and the model starts predicting fake torque while you're relaxed. Labelling them properly as 0 (E22) does the opposite.

---

## 5. What I tested (and why)

First a literature search: 4 topics, 67 sources, every source re-checked by a second agent that opened the paper (section 13). It ranked the ideas, and I then tested every one that applies to this data. The literature only *suggests* these; none of the papers tested this exact setup. "Base" = what the change is applied to.

| # | Change | Base | Question | Source / idea |
|---|---|---|---|---|
| E01 | — | — | old pipeline | — |
| E02 | report-1 minimal fix | — | reference | report 1 |
| E03 | report-1 final pipeline (passive gain fix + EMG auto-gain + mixture-of-experts LSTM + per-angle scaled loss) | — | reference | report 1 |
| E04 | passive = the hold recorded just before *this* trial | E03 | your hold idea | your suggestion |
| E06 | passive that relaxes during the trial (log-shaped drift) | E03 | 〃 | stress relaxation: Duong 2001; McNair 2001; Tian 2010 |
| E10 | EMG ÷ MVC peak (global) | E03 | your MVC idea | Burden 2010 |
| E11 | EMG ÷ MVC peak at this angle | E03 | 〃 | Hali 2020 |
| E13 / E14 | loss ×3 / ×10 on the 2 most plantarflexed angles | E03 | your "10× error" idea | — |
| E16 | random EMG gain during training (σ 0.5) | E03 | robustness to EMG level changes | Tsinganos 2020 (augmentation) |
| E18 | 〃 | E02 | 〃 | 〃 |
| E19 | original small LSTM + auto-gain | E02 | is the big model needed? | — |
| E20 / E21 | pre-train on the 5 other subjects, then fine-tune | E19 / E02 | transfer learning | Zhang L 2022; Xie 2025; Wang 2024 |
| E22 | rest-anchored training labels | E03 | cleaner labels | section 3 (in-session passive: Kay & Blazevich 2009; Hoang 2005) |
| E23 | rest-anchored labels, no auto-gain | E02 | 〃 | 〃 |
| E24 / E25 | relaxed windows removed / ×0.2 | E03 | your question | section 4 |
| E26 | each session scaled by its own EMG level (uses unlabelled test EMG) | E03 | per-session normalisation | Du 2017 (AdaBN idea) |
| E27 / E28 | ridge regression on lagged EMG (+ squares), one coefficient set per angle | — / + auto-gain | is deep learning needed? | Ziai 2011; Clancy 2012 |
| E29 | 400 ms window instead of 200 ms | E03 | longer memory | Wang 2024 |
| E31 | small LSTM + auto-gain + rest-anchored labels | E19 | simplest good candidate | (added after the first results, see 6) |

**Defined but not run, on your request** (cut when you trimmed the plan, as likely-redundant variants; note the 600 ms window was cut *before* the 400 ms window had run, and the 400 ms window later gave a small gain): hold-end level, hold-curve extrapolation, minimal+drift, softer auto-gain, σ 0.2 augmentation, augmentation without auto-gain, no-auto-gain ablation, and the 600 ms window. They're in `ML/research/experiments.py` (E05, E07, E08, E09, E12, E15, E17, E30) if you ever want them.

---

## 6. How I judged it (without cheating)

- **Main score:** for each test trial (= one angle of one subject in one direction), R² against the reference truth, first averaged over the 3 seeds and then **floored at −1**, so one catastrophic angle can't swamp everything. Then the mean over the 8 angles, then over subjects × directions. I also report RMSE (Nm), RMSE as % of the MVC at that angle, R² on the pushing parts only, and phantom torque at rest.
- **Picking the winner without peeking.**
  - The rule: highest main score on YES/JM/EG with 3 seeds.
    - It's written down in `ML/research/results/SELECTION_RULE.md`, with timestamps. I wrote it **after most screening results were in, but before I opened any held-out result**.
    - Many held-out runs had already finished in the background by then (their logs print scores), but none had been opened. You only have my word for that last part.
  - **Cross-fitting:** pick the best configuration on one direction, score it on the other. That gives an honest estimate of "pick the best, then use it".
  - **Leave-one-subject-out:** pick on 2 subjects, score on the third.
- **Statistics:** paired comparisons per test trial (Wilcoxon test + bootstrap 95% CI). With ~23 comparisons I use Holm-corrected p-values. With 3 subjects this describes *these* subjects, not a population.
- **Configurations added along the way, flagged so you can judge:**
  - **E31** was added after I'd seen the first results. It fills the missing cell of the 2×2 grid {passive fix: gain / rest-anchored} × {model: small LSTM / mixture-of-experts}.
  - **E29** first ran with 1 seed. It was better than E03 on the same seed in 6 / 6 cells, so I gave it the 2 other seeds, which made it eligible. That decision used screening data only.
  - **E23 and E32 on CH** are small diagnostics, run *after* the held-out check and on CH only (section 9). They were not eligible.
- **E26 is "transductive":** it needs the whole test session's EMG before it can predict, so it wasn't allowed to win.
- **Independent checks.** Before the main runs, a separate set of agents reviewed the code for leakage and wrong metrics. 16 issues were confirmed, all minor or moderate; all were fixed and the affected runs redone (old results kept in `results/superseded_val_weights/`). After the report was written, another set re-checked every number in it against the result files.

---

## 7. Results

![All configurations](figures/fig03_all_configs.png)

*Figure 4: every configuration, sorted by score. Each dot = one subject × direction (mean of 3 seeds), black bar = mean.*

### Main table (YES, JM, EG × 2 directions × 3 seeds = 18 models per row)

"weak-2" = the 2 most plantarflexed angles of each subject. "phantom" = mean |prediction| while relaxed.

| # | configuration | R² | R² weak-2 | RMSE weak-2 (Nm) | RMSE all (Nm) | RMSE (% MVC) | phantom (Nm) |
|---|---|---|---|---|---|---|---|
| E01 | old pipeline | 0.540 | 0.04 | 1.27 | 2.04 | 11.2 | 1.25 |
| E02 | report-1 minimal fix | 0.826 | 0.64 | 0.76 | 1.34 | 4.9 | 0.21 |
| **E29** | **E03 + 400 ms window** (picked by the rule) | **0.932** | 0.84 | 0.44 | **0.82** | **2.9** | 0.24 |
| **E03** | **report-1 final** | 0.925 | **0.85** | 0.44 | 0.90 | 3.2 | 0.25 |
| **E22** | **E03 + rest-anchored labels** (recommended, section 10) | 0.925 | 0.84 | 0.45 | 0.92 | 3.2 | **0.17** |
| E24 | E03, relaxed windows removed | 0.924 | 0.84 | 0.45 | 0.92 | 3.2 | 0.36 |
| E25 | E03, relaxed windows ×0.2 | 0.923 | 0.84 | 0.45 | 0.92 | 3.2 | 0.29 |
| E16 | E03 + EMG gain augmentation | 0.922 | 0.83 | 0.42 | 0.90 | 3.1 | 0.25 |
| E13 | E03 + ×3 weight plantarflexed | 0.922 | 0.83 | 0.45 | 0.93 | 3.3 | 0.25 |
| E04 | E03 + passive from own before-hold | 0.921 | 0.82 | 0.45 | 0.90 | 3.2 | 0.24 |
| E06 | E03 + passive relaxation drift | 0.921 | 0.83 | 0.44 | 0.91 | 3.2 | 0.25 |
| E19 | small LSTM + auto-gain | 0.917 | 0.83 | 0.47 | 1.01 | 3.4 | 0.22 |
| E31 | small LSTM + auto-gain + rest-anchored | 0.917 | 0.84 | 0.50 | 1.04 | 3.6 | **0.15** |
| E14 | E03 + ×10 weight plantarflexed | 0.916 | 0.83 | 0.46 | 1.01 | 3.5 | 0.25 |
| E11 | E03 with EMG ÷ MVC (per angle) | 0.904 | 0.83 | 0.55 | 1.08 | 3.8 | 0.25 |
| E10 | E03 with EMG ÷ MVC (global) | 0.887 | 0.81 | 0.57 | 1.11 | 4.0 | 0.26 |
| E20 | E19 + pre-train on 5 other subjects | 0.885 | 0.72 | 0.51 | 1.12 | 3.7 | 0.22 |
| E28 | ridge regression + auto-gain | 0.863 | 0.80 | 0.60 | 1.53 | 4.7 | 0.42 |
| E18 | E02 + EMG gain augmentation | 0.857 | 0.76 | 0.53 | 1.17 | 4.5 | 0.33 |
| E23 | small LSTM + rest-anchored (no auto-gain) | 0.853 | 0.76 | 0.69 | 1.37 | 4.9 | 0.12 |
| E21 | E02 + pre-train on 5 other subjects | 0.816 | 0.67 | 0.75 | 1.45 | 5.2 | 0.23 |
| E26 | per-session EMG scale (transductive, not eligible) | 0.775 | 0.84 | 0.45 | 1.43 | 5.5 | 0.50 |
| E27 | ridge regression | 0.624 | 0.54 | 0.81 | 2.85 | 9.8 | 0.43 |

### What the selection rule says (screening subjects only)

| | value |
|---|---|
| Best on S1→S2 / best on S2→S1 | E29 / E29 |
| Cross-fitted score (pick on one direction, score on the other) | 0.932. This equals the plain best score because E29 won both directions. It is optimistic: E29 only got its 2 extra seeds after its seed-0 run had beaten E03 in both directions |
| Leave-one-subject-out (pick on 2 subjects, score the 3rd) | 0.925 |
| Per-angle R² below 0 (seed mean, 48 test trials) | old: **8 / 48**; E03, E22, E29, E31, E19: **0 / 48** |
| Seed spread of the main score (std over 3 seeds) | E03: 0.001–0.009, E29: 0.001–0.014 depending on subject (very stable) |
| Seed ensemble (average the 3 networks' predictions) | E03: 0.931, E29: 0.938 (≈ +0.005–0.006 for free) |

### Is it real? (descriptive paired comparisons)

These use 48 per-angle results, but they come from only 3 people, so they aren't independent. The p-values describe these 3 subjects, not people in general. Holm-corrected unless marked.

| comparison | mean difference in per-angle R² | 95% CI | better in | p |
|---|---|---|---|---|
| E29 vs old, all angles (cross-fitted) | **+0.392** | +0.232 to +0.568 | 48 / 48 | < 10⁻¹² (uncorrected) |
| E29 vs old, 2 most plantarflexed (cross-fitted) | **+0.796** | +0.474 to +1.134 | 12 / 12 | 0.0005 (uncorrected; Holm 0.011) |
| E03 vs old, 2 most plantarflexed | +0.802 | +0.473 to +1.154 | 12 / 12 | 0.011 |
| E29 vs E03, all angles | +0.007 | +0.001 to +0.013 | 33 / 48 | 0.055 (Holm) |
| E29 vs E03, 2 most plantarflexed | −0.006 | −0.020 to +0.009 | 2 / 12 | n.s. |
| E22 vs E03 | −0.001 | −0.010 to +0.009 | 21 / 48 | 1 (no difference) |
| E31 vs E03 | −0.009 | −0.020 to +0.003 | 10 / 48 | 0.015 (slightly worse) |
| ×10 plantarflexed weight (E14) vs E03 | −0.009 | −0.017 to −0.004 | 12 / 48 | 0.0025 (worse) |
| pre-training (E20) vs E03 | −0.041 | −0.107 to −0.004 | 12 / 48 | 0.0065 (worse; vs its own control E19 the mean is −0.032) |

### What each idea did, in plain words

- **EMG auto-gain is the big one.** Adding it to the original tiny LSTM (E02 → E19) goes 0.826 → 0.917. The MVC normalisations are worse than auto-gain. The global version (E10: −0.039, Holm p = 0.011) is clearly worse. The per-angle version (E11: −0.021) is lower but not significantly so after correction. The MVC was only recorded once (at the start of the visit), and the EMG size changes between the two sessions (report 1: up to ~0.4× for JM's calf muscles), so a fixed MVC scale can't follow that change, while auto-gain adapts as it goes (using only past EMG).
- **The mixture-of-experts LSTM adds a little.** It gives about +0.008 (E19 → E03, E31 → E22), and it avoids the occasional overshoot of the small LSTM (Figure 7d).
- **A 400 ms window adds a little** (+0.007, mostly at the strong angles). The plantarflexed angles don't change.
- **Your holds (E04/E06) tie E03.** They only change the passive estimate in the *training* labels, and E03's passive fix was already close to the in-trial truth for these 3 subjects.
- **Extra weight on the plantarflexed angles doesn't help.** ×3 ties. ×10 is worse overall, and it doesn't even improve the plantarflexed angles (0.83 vs 0.85). E03's per-angle scaling already gives every angle the same importance.
- **Augmentation** helps the simple model without auto-gain (E02 → E18: +0.03) but adds nothing on top of auto-gain. They fix the same problem (EMG level changing between sessions), so you only need one.
- **Pre-training on the other 5 subjects hurts** (E19 → E20: −0.03; YES drops to 0.71 in one direction). The subjects differ too much (angles, strength, and for CH even push direction).
- **Ridge regression is far behind** (0.62, or 0.86 with auto-gain). So the LSTM *does* add something, but auto-gain matters more than the model.
- **Per-session EMG scaling (E26)** works for YES and EG but breaks JM (0.30 in one direction), and it needs the whole test session in advance.

![Per-angle R² heatmap](figures/fig05_per_angle_heatmap.png)

*Figure 5: per-angle R² (mean of 3 seeds) for the main configurations. Columns = subject, test session, angle. Red = bad. The old pipeline has red cells at the plantarflexed angles of every subject. The remaining weak spot of the new LSTM models (E03, E19, E22, E29, E31) is YES −0.49 in session 2 (R² 0.3–0.5). With pre-training (E20) that cell collapses to −1.*

### Short labelled calibration (bonus)

If each new session started with a few seconds of labelled data, a linear correction (a·prediction + b) could be fitted per trial, a common trick in the literature (Vidovic 2016; Lloyd & Besier 2003). Everything is scored on t ≥ 30 s, so every calibration length is compared on the same samples.

![Calibration](figures/fig07_calibration.png)

*Figure 6: with the new pipelines, 5–10 s of calibration makes things **worse** (the correction overfits the first seconds), and 20–30 s adds only ~+0.01–0.02 for E03 / E22 / E29. For the small LSTM (E31), even 20 s is slightly worse than none, and only 30 s helps (+0.01). The old pipeline needed calibration badly (0.57 → 0.90). So the new pipeline already removes most of the session offset that calibration used to patch.*

---

## 8. The plantarflexed angles up close

R² per angle against the reference truth, mean of 3 seeds. The [appendix](appendix_tables.md) has every angle, and RMSE too.

**YES**

| test direction | angle | old | E03 | E29 (400 ms) | E22 (rest-anchored) | E31 (small) |
|---|---|---|---|---|---|---|
| S1 → S2 | −0.486 | −14.28 | 0.47 | 0.42 | 0.34 | 0.55 |
| S1 → S2 | −0.381 | 0.14 | 0.84 | 0.83 | 0.82 | 0.83 |
| S2 → S1 | −0.486 | −7.55 | 0.72 | 0.70 | 0.70 | 0.74 |
| S2 → S1 | −0.382 | 0.85 | 0.93 | 0.93 | 0.91 | 0.87 |

**JM**

| test direction | angle | old | E03 | E29 | E22 | E31 |
|---|---|---|---|---|---|---|
| S1 → S2 | −0.787 | 0.18 | 0.90 | 0.90 | 0.91 | 0.89 |
| S1 → S2 | −0.644 | 0.50 | 0.88 | 0.94 | 0.91 | 0.85 |
| S2 → S1 | −0.787 | −0.99 | 0.76 | 0.75 | 0.90 | 0.89 |
| S2 → S1 | −0.645 | −0.44 | 0.78 | 0.74 | 0.79 | 0.61 |

**EG**

| test direction | angle | old | E03 | E29 | E22 | E31 |
|---|---|---|---|---|---|---|
| S1 → S2 | −0.739 | −0.09 | 0.96 | 0.95 | 0.96 | 0.93 |
| S2 → S1 | −0.739 | 0.68 | 0.94 | 0.96 | 0.92 | 0.95 |

**In Nm, YES −0.486 trained on S1, tested on S2:** RMSE goes from 1.52 Nm (old) to 0.26–0.31 Nm (the new models), i.e. from 11% to ~2% of the MVC at that angle (~13.7 Nm, interpolated). R² is still only 0.3–0.55 there.
- The person pushes only ~1 Nm at that angle.
- The remaining error (~0.3 Nm) is still about a third of that push, and close to the signal's own variation (SD ≈ 0.4 Nm).

So it's small compared with the MVC, but relative to what the person actually does at this angle, the push isn't tracked well yet. Whether 0.3 Nm is good enough depends on what you use the model for.

![YES S1 to S2, plantarflexed angles](figures/fig06a_overlay_YES_S1toS2.png)

*Figure 7a: YES, trained on session 1, tested on session 2. Black = reference truth. The old pipeline (blue) follows the timing of the pushes but sits ~1.5 Nm too low; that constant offset is what drives its R² down to −14. All new models follow the pushes. At −0.381 they all overshoot the deep pushes a bit (mean bias about −0.6 Nm); that's the main error left.*

![YES S2 to S1](figures/fig06b_overlay_YES_S2toS1.png)
![JM S1 to S2](figures/fig06c_overlay_JM_S1toS2.png)
![JM S2 to S1](figures/fig06d_overlay_JM_S2toS1.png)

*Figures 7b–d: same idea for YES S2 → S1 and for JM in both directions. In 7d (JM −0.645), the small LSTM (E31) has a few big overshoot spikes: −4.5 Nm where the truth is −2. The mixture-of-experts models (E03 / E22 / E29) don't, which is the reason to keep the slightly bigger model.*

![EG S1 to S2](figures/fig06e_overlay_EG_S1toS2.png)
![EG S2 to S1](figures/fig06f_overlay_EG_S2toS1.png)

**Full-trial inspection plots (all 8 angles) exist for every configuration, subject and direction.** They're saved next to each result, in `ML/research/results/<screen|confirm>/<subject>/<direction>/<config>/`:

- `full_trial_inspection_truth_<S1|S2>_fast_seed0.png`: against the reference truth
- `full_trial_inspection_<S1|S2>_fast_seed0.png`: against the configuration's own target, the same style as before

These exist for the seed-0 run of every screening and held-out configuration; seeds 1 and 2 have no plots.

---

## 9. Check on 3 subjects held out from this round's choices

IES, CH and HM were not scored or looked at for any decision of this round before the selection. They aren't fully fresh data, though:
- all three were analysed in report 1;
- CH's push-direction flip shaped report 1's design (the scale-only mixture-of-experts head and the auto-gain `auto` switch);
- their recordings were pre-training data for E20 and E21.

IES is the cleanest of the three. Five configurations ran on them (seed 0, both directions): the old pipeline, E03, the rule's pick E29, and the two rest-anchored candidates E22 and E31.

| held-out subjects | R² (all angles) | R² weak-2 | RMSE weak-2 (Nm) | phantom (Nm) | R², without CH's flipped angle* |
|---|---|---|---|---|---|
| old pipeline | 0.874 | 0.70 | 1.23 | 1.07 | 0.878 |
| E03 | 0.787 | 0.66 | 1.32 | 0.85 | 0.870 |
| E29 (rule's pick) | 0.797 | 0.67 | 1.32 | 0.85 | 0.880 |
| **E22** | **0.884** | **0.97** | **0.49** | **0.18** | **0.967** |
| E31 | 0.869 | 0.94 | 0.59 | 0.19 | 0.951 |
| **E22 + "auto" rule** (auto-gain off for CH only; post-hoc, see below) | **0.966** | **0.97** | **0.48** | **0.17** | — |

*\*Post-hoc, explained below.*

![Held-out subjects, plantarflexed angles](figures/fig08b_untouched_weak.png)

*Figure 8: R² at the 2 most plantarflexed angles of each held-out subject (seed 0, mean of both directions). HM: old 0.57 → ~0.94 for E03 / E29 / E22 (E31: 0.86). IES: the pipelines that need a passive recording (E03, E29) collapse to ~0.05–0.08, while the rest-anchored ones (E22, E31) reach 0.98. All-angle version: `figures/fig08a_untouched_all.png`.*

Per subject (mean of the 2 directions):

| | CH | HM | IES |
|---|---|---|---|
| old | 0.961 | 0.840 | 0.821 |
| E03 | 0.728 | 0.937 | 0.697 |
| E29 | 0.736 | 0.950 | 0.705 |
| E22 | 0.739 | 0.930 | **0.983** |
| E31 | 0.731 | 0.895 | 0.980 |

**What happened, subject by subject:**

- **HM behaves like the screening subjects.** Its most plantarflexed angle (−0.789) goes from 0.08 / 0.43 (old) to 0.89–0.93 with E03, E29 and E22. The small LSTM (E31) reaches 0.92 in one direction but only 0.63 in the other.
- **IES has no passive recordings at all.** No static hold and no passive map exist in either session.
  - Pipelines that need one (old, E03, E29) then subtract only a constant, and the angle-dependent passive torque (several Nm) stays in their training labels.
  - E03 and E29 end up with a bias of up to ~3 Nm and R² down to −0.5 at IES's plantarflexed angles, in both directions.
  - The old pipeline also has no passive estimate for IES, yet it does fine in one direction (0.90 / 0.94 trained S1 → tested S2) and fails only in the other (−0.11 / 0.59). So the missing passive recording doesn't explain the E03 / E29 collapse on its own.
  - My guess (not tested): E03's mixture-of-experts head has no per-angle offset, by design from report 1, so it can't absorb a passive torque that changes with angle. The old LSTM can partly absorb it through its angle input.
  - The rest-anchored labels (E22, E31) don't need any passive recording, because they measure the passive level from the relaxed moments of each trial: **R² 0.97–0.99 at every angle**.
  - That makes rest-anchored labels the more robust choice when passive recordings are missing. But it rests on one subject. The direct test would be to re-run E03 vs E22 on YES / JM / EG with their passive recordings removed.
- **CH is excellent except at one angle (−0.31 / −0.33 rad).** At its other angles E22 gets 0.98–1.00 and E29 / E31 get 0.96–0.99. (Smaller dips: the old pipeline gets 0.94–0.95 at its plantarflexed angles in one direction, and E03 gets 0.82 at −0.456 in one direction.)
  - At that angle CH pushed **toes-up** (+28 Nm, tibialis anterior) in session 1 and **toes-down** (−28 Nm, calf) in session 2, so the task itself changed between sessions.
  - Every auto-gain model fails there (R² −1.5 to −5.7). The old pipeline gets 0.97 / 0.79 at that angle.
  - Since the angle is floored at −1, it alone pulls CH's mean down from ~0.98 to ~0.73.
  - The right-hand column above drops that single angle. That's a post-hoc choice, made because the task wasn't the same.
- **What causes the CH failure?** I ran two diagnostics, both after unblinding and on CH only:
  - **E32 = E22 with the auto-gain switched off.** It gets CH right everywhere, including the flipped angle: R² **0.98 / 0.95** there and **0.988 / 0.983** overall. E22 gets −3.3 / −5.5 at that angle.
  - **E23 = rest-anchored labels + the small LSTM, no auto-gain.** It fixes the flipped angle in one direction only (0.98 trained S1 → tested S2, but 0.05 the other way).
  - So the **auto-gain** is what breaks the direction flip. Removing it is enough with the mixture-of-experts head (E32), but not with the small LSTM (E23). Auto-gain rescales each muscle to its own recent range, so the model loses which muscle (calf or shin) is really doing the work.
  - Report 1 already had a rule for this, written before this round: the **"auto" mode** turns auto-gain off when the training session contains pushes in both directions.
  - Among the 6 subjects, that rule switches auto-gain off **only for CH**. For YES, JM, EG, IES and HM nothing changes.
  - **E22 + auto rule on the held-out subjects: R² 0.966, plantarflexed R² 0.97, phantom torque 0.17 Nm** (CH 0.986, HM 0.930, IES 0.983). This number is post-hoc, since CH is scored with the E32 diagnostic run.
  - It also has **no matched comparison**: E03 and E29 weren't re-run with the same rule.
  - The like-for-like comparison is the right-hand column of the table above: all 5 pipelines on the same test trials, only CH's flipped angle left out. There it's E22 **0.967**, E31 0.951, E29 0.880, old 0.878, E03 0.870.

> **Honesty note:**
> - The selection rule picked E29. On all angles of the held-out subjects, E29 doesn't beat the old pipeline; IES and CH explain why.
> - I recommend E22 below *because of* this check, so the check is no longer independent evidence for E22 itself. Only a new subject (or new sessions) could confirm it cleanly.
> - The switch rests mostly on IES, one subject. The exact mechanism isn't isolated (see above).
> - On CH and HM, E22, E03 and E29 are about equal at the plantarflexed angles.

---

## 10. What I'd use

**Recommended: E22 = the report-1 final pipeline with rest-anchored training labels, with report 1's auto-gain `auto` rule.** It's available as one configuration, **`FINAL`** (`--config FINAL`). I ran it on all 6 subjects in both directions (seed 0), and it gives exactly the same predictions as E22 (E32 for CH) in all 12 runs. Step-by-step explanation: [final_code_pipeline.md](../final_pipeline/final_code_pipeline.md). In order:

1. **Rest-anchored labels:** target = measured torque − a smooth curve through the torque during the EMG-detected relaxed moments of the same trial (training session only). This needs **no passive recording** and follows the in-trial drift.
2. **EMG auto-gain:** causal, so it adapts to the EMG level of a new session. It is switched off automatically when a subject pushes in both directions (report 1's `auto` rule; only CH).
3. **Mixture-of-experts LSTM:** one small correction per training angle.
4. **Per-angle scaled loss:** every angle counts the same.

Why E22 over E03 / E29:
- On YES / JM / EG the plantarflexed angles are the same (0.84 vs 0.85 / 0.84; no significant difference), and E22 has the least phantom torque (0.17 vs 0.25 Nm).
- On the held-out subjects it doesn't break when a subject has no passive recording (IES 0.983), unlike E03, E29 and the old pipeline; only the two rest-anchored configurations hold up there (E22, E31). Plantarflexed R² on the held-out subjects: 0.97 vs 0.66–0.70.

| If you want… | use |
|---|---|
| the most robust choice (recommended) | **E22** |
| the best score on YES / JM / EG (the pre-set rule) | E29 (E03 + 400 ms window), but only if every subject has clean passive holds |
| the simplest thing that almost works as well | **E31** (original tiny LSTM + auto-gain + rest-anchored labels): −0.009 R² on YES / JM / EG, but clearly weaker on held-out HM (0.895 vs 0.930), occasional overshoots, and its version without auto-gain doesn't fix CH's flip in both directions |
| a free +0.006 | average the 3 seeds' models (ensemble) |
| a subject who pushes in **both directions** (like CH) | E22 with auto-gain off. Report 1's `auto` rule does this by itself: auto-gain goes off when the training session has pushes in both directions. Only CH triggers it |

**Cheap next step:** E22 with a 400 ms window. It combines the two best ideas, but I haven't tested it, so I'm not claiming it.

**What to stop worrying about:**
- Holds and drift models: no gain once the labels are anchored in-trial.
- Extra plantarflexed weighting: no gain.
- MVC normalisation: worse than auto-gain.
- Augmentation on top of auto-gain: no gain.
- Pre-training on other subjects: worse.

---

## 11. Limitations

- **3 study subjects + 3 for the check.** Everything describes these 6 people; the p-values aren't population claims. The held-out subjects got 1 seed each, and they aren't fully fresh (report 1 analysed them, and they were pre-training data for E20 / E21).
- **The final recommendation (E22) was informed by the held-out check** (section 9), and mostly by one subject (IES). Only new data can confirm it independently.
- **Circularity:** E22's training labels use the same construction as the scoring truth (section 3), so its phantom-torque advantage is partly built in.
- **Selection rule timing:** the rule was written after most screening results were in (but before the held-out results were opened), and E29 got its extra seeds after its seed-0 result was seen. The cross-fitted 0.932 is therefore optimistic; leave-one-subject-out gives 0.925.
- **YES −0.486 tested on session 2 is still the weakest cell** (R² 0.34–0.55 depending on the model). The person only produces ~1 Nm there, so a few tenths of a Nm of error caps R². In Nm it's fine (~0.3 Nm), but the R² will always be fragile there.
- **The reference truth is a definition, not a perfect measurement.**
  - It assumes torque during EMG-detected rest is purely passive.
  - When someone is rarely fully relaxed in a trial (e.g. JM S2 −0.787: only 4% of samples detected as rest), the baseline is less certain.
  - If the muscle fibres shorten during a push, the passive contribution could change during pushes (MacIntosh 2005; Hoffman 2012). That can't be measured here without ultrasound.
- **Cross-fitting shares subjects.** Picking on one direction and scoring on the other still uses the same 3 people. The leave-one-subject-out estimate (0.925) and the held-out subjects are the less optimistic checks.
- **The retest-in-training ("pooled") question** wasn't re-run (your call). Report 1's answer still stands: it's good for a final model, but measure it with cross-session tests.
- **Fast training setting.** All screening used batch 64. Report 1 showed the old-vs-new conclusion holds with your original batch-8 setting; I didn't redo that check this round.
- **CPU only.** TensorFlow has no GPU support on native Windows, and porting to PyTorch would have changed the implementation being compared. The models are tiny anyway (1k–4k parameters).

---

## 12. The code

All new code is in **`ML/research/`**. It imports the existing pipeline and **does not modify it**. Every module starts with a docstring explaining what it does and why.

| file | lines | what it does |
|---|---|---|
| `data.py` | 201 | loads a subject: active trials of both sessions, the hold before each trial, MVC trials (cached) |
| `truth.py` | 146 | the common reference truth + rest / active metrics |
| `passive.py` | 221 | passive models (old map, static, gain, hold-based, drift) |
| `emg_norm.py` | 135 | EMG normalisations (train max, auto-gain, MVC, per-session) |
| `splits.py` | 85 | cross-session folds, pooled split, windows |
| `models.py` | 72 | LSTM / mixture-of-experts + EMG gain augmentation layer |
| `train.py` | 190 | training, per-angle loss balancing, relaxed-window weighting, ridge baseline |
| `transfer.py` | 144 | pre-training on the other subjects + fine-tuning |
| `experiments.py` | 109 | the named configurations, each with the question it answers |
| `run.py`, `queue.py` | 266 | run one job / many in parallel (skips finished or running jobs) |
| `analyze.py`, `tables.py`, `figures.py` | ~920 | tables, cross-fitted selection, statistics, figures |

About **2,600 lines in total**. Run instructions are in `ML/research/README.md`, for example:

```bash
source ML/experiments/env.sh
$PY -m ML.research.queue --study screen --subjects YES JM EG --protocols S1toS2 S2toS1 --configs E01_old E22_restanchor_final --seeds 0 1 2
$PY -m ML.research.analyze --study screen
```

**Results.**
- Screening runs are in `ML/research/results/screen/`, the held-out check in `confirm/`.
- Each run has a `.json` (config, passive coefficients, metrics) and a `.npz` (predictions). The seed-0 runs also have both trial inspection plots.
- Summaries are in `results/<study>/_summary/summary.json` and `tables.md`.

**Bugs found and fixed during the round** (all caught by the independent code review before the main runs):
- The analysis read its own cache files and crashed on its second run.
- The weighted-loss configurations used validation weights that didn't match training (E02 was re-run).
- The ridge regularisation grid was too narrow.
- The pooled EMG scale could see test-part EMG; this never affected a reported number, because the pooled runs were skipped.

---

## 13. Sources

Every source was opened and checked by a second agent: title, authors, year, DOI, and that the claim is actually in the paper. **None of them tested this exact setup** (an LSTM, isometric multi-angle plantarflexion, cross-session). They justify *trying* an idea; my own runs decide whether it works here.

**Ideas that worked or were kept**
- **Passive torque changes with history and between sessions, so it must be measured in-session.** This is the basis for the reference truth and the rest-anchored labels.
  - Kay & Blazevich 2009, *J Appl Physiol* 107 — [PubMed 19644033](https://pubmed.ncbi.nlm.nih.gov/19644033/)
  - Hoang et al. 2005, *J Biomech* 38(6) (between-session passive differences)
- **The ankle's stress relaxation during holds.** This is why the hold level ≠ the in-trial level.
  - Duong et al. 2001, *Clin Biomech* 16(7) — doi:[10.1016/S0268-0033(01)00043-2](https://doi.org/10.1016/S0268-0033(01)00043-2)
  - McNair et al. 2001, *Med Sci Sports Exerc* 33 — [PubMed 11252058](https://pubmed.ncbi.nlm.nih.gov/11252058/)
  - Tian et al. 2010, *J Biomech* 43 — [PubMed 20003980](https://pubmed.ncbi.nlm.nih.gov/20003980/)
- **EMG amplitude changes between recordings (e.g. when electrodes are re-applied), so it needs per-session normalisation.** This is why auto-gain matters. (In our data the two sessions are two blocks of the same visit and recording file, so I can't say whether the electrodes were re-applied, but the EMG size did change between them.)
  - Burden 2010, *J Electromyogr Kinesiol* 20(6) — doi:[10.1016/j.jelekin.2010.07.004](https://doi.org/10.1016/j.jelekin.2010.07.004)
  - Schulte et al. 2022, *Front Robot AI* 9:869476 — doi:[10.3389/frobt.2022.869476](https://doi.org/10.3389/frobt.2022.869476) (deep EMG→torque models degrade across days even with daily MVC normalisation)
- **Longer input windows (~300–600 ms):** Wang H et al. 2024, *Sensors* 24:7301 — doi:[10.3390/s24227301](https://doi.org/10.3390/s24227301). A small gain here (E29).

**Tested here, did not help**
- **Pre-training + fine-tuning:**
  - Zhang L et al. 2022, *IEEE TNSRE* 30 — doi:[10.1109/TNSRE.2022.3156786](https://doi.org/10.1109/TNSRE.2022.3156786)
  - Xie et al. 2025, *Front Bioeng Biotechnol* — doi:[10.3389/fbioe.2025.1530950](https://doi.org/10.3389/fbioe.2025.1530950)
  - Wang H 2024 (above)
- **EMG amplitude augmentation:** Tsinganos et al. 2020, *Sensors* 20:4892 — doi:[10.3390/s20174892](https://doi.org/10.3390/s20174892) (a classification study, not regression)
- **Unsupervised per-session re-normalisation (AdaBN idea):** Du et al. 2017, *Sensors* 17:458 — doi:[10.3390/s17030458](https://doi.org/10.3390/s17030458)
- **Linear / polynomial EMG→torque baselines:**
  - Ziai & Menon 2011, *J NeuroEng Rehabil* 8:56 — doi:[10.1186/1743-0003-8-56](https://doi.org/10.1186/1743-0003-8-56)
  - Clancy et al. 2012, *IEEE TBME* 59(1) — doi:[10.1109/TBME.2011.2170423](https://doi.org/10.1109/TBME.2011.2170423)
- **Angle-specific MVC normalisation:** Hali et al. 2020, *Physiol Rep* 8:e14680 — [Europe PMC PMC7757371](https://europepmc.org/article/PMC/PMC7757371)

**Background for the limitations**
- **Short labelled recalibration:**
  - Vidovic et al. 2016, *IEEE TNSRE* — doi:[10.1109/tnsre.2015.2492619](https://doi.org/10.1109/tnsre.2015.2492619)
  - Lloyd & Besier 2003, *J Biomech* 36(6) — doi:[10.1016/S0021-9290(03)00010-1](https://doi.org/10.1016/S0021-9290(03)00010-1)
- **Passive subtraction during contraction is model-dependent:** MacIntosh & MacNaughton 2005, *J Appl Physiol* 98; Hoffman et al. 2012, *J Appl Physiol* 113

The full verified list (67 sources) and the literature synthesis are kept with the session notes.

---

## Appendix

- **All tables for the screening subjects:** [appendix_tables.md](appendix_tables.md). The overview includes the own-target R² for comparison with report 1, plus per-angle R² and RMSE for every subject and direction.
- **The same for the held-out subjects:** [appendix_tables_confirm.md](appendix_tables_confirm.md)
- **Reference-truth inspection for all 6 subjects:** `figures/truth_inspection_<subject>.png`
- **Machine-readable results:** `ML/research/results/screen/_summary/summary.json` and `.../confirm/_summary/summary.json`
- **Selection rule, written down before unblinding (with timestamps):** `ML/research/results/SELECTION_RULE.md`
