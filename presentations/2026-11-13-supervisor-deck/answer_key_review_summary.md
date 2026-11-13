# Is our "in-trial passive" answer key circular? Summary for review

## 1. Setup
- **Goal:** predict voluntary ankle torque from 4 surface EMG channels plus the ankle angle.
  - Channels: GM, GL and SOL (calf, push toes down) and TA (shin).
  - Model: a small LSTM, one model per person.
- **Experiment (isometric):**
  - The foot is locked at 8 angles, from −0.79 to +0.25 rad (negative = toes down).
  - One 90 s trial per angle: the person follows an on-screen push/relax target (0–40 % of MVC).
  - 6 people, 2 sessions each, in the same visit with the electrodes left on.
- **Training and testing:**
  - Fit on one session: the first 80 % of each trial is used for fitting, the last 20 % for early stopping.
  - Test on the other session, and also in the opposite direction.
- **Scores:** R² and RMSE per angle.

## 2. The core issue
The torque sensor measures total torque = passive + voluntary. Voluntary torque is not measured directly; you get it by subtracting a passive estimate. That estimate defines both the training labels and the target used for scoring.

## 3. Old passive estimate (last semester)
**Method:**
- Take the separate relaxed recording made before each pushing trial at the same angle. Only its first ~20 s are a clean, still, relaxed hold.
- Build a table of one value per angle, and subtract one constant per pushing trial.

**Problems found:**
- **The table is higher than the relaxed level inside the pushing trial.**
  - In-trial level ≈ 81 % of the table value (median ratio).
  - Median absolute difference: 0.58 Nm over 80 trials.
  - At the 2 most toes-down angles of each person, the table was too high in 20 of 20 trials (median +0.64 Nm, max +2.03 Nm). Pushes there are only about 1 Nm.
- **Drift within a trial:** median 0.15 Nm, but 0.5 Nm at YES −0.49 rad.
- **One bug:** JM's table had a +11.7 Nm entry built from segments that were not relaxed. That gave a 13.7 Nm error at −0.08 rad.

## 4. New passive estimate ("rest-anchored")
1. **Find rest moments from EMG only.**
   - Activity = the mean over the 4 channels of (envelope − trial floor) ÷ (session range).
   - Rest starts when activity drops below 3 % and ends when it rises above 6 %.
   - A rest stretch must last at least 0.5 s; its first 0.3 s and last 0.1 s are dropped.
2. Take the median torque of each rest stretch.
3. Smooth these medians with a Gaussian kernel (σ = 8 s), with outlier rejection and flat ends. This gives the baseline B(t).
4. Voluntary torque = measured torque − B(t).

**Where it is used:**
- As the **training labels** (training session only).
- As the **shared answer key** used to score every model, old and new, on the test session.
- The model never receives B(t) as an input; it only sees EMG and angle.

## 5. The concern
The new pipeline's training labels and the answer key are built with the same recipe, which risks circularity. The key is also a definition, not a measurement: it assumes that quiet EMG means zero voluntary torque.

## 6. Evidence so far
**Agreement with the separate recordings.**
- The in-trial relaxed level follows the same pattern with angle, with the same sign and size.
- It is systematically about 19 % lower. That is consistent with stress relaxation and thixotropy (Duong 2001; McNair 2001; Kay & Blazevich 2009).

**A comparator whose labels come from a different recipe:**
- Our earlier pipeline built its labels from the separate passive recordings with a correction (α·P + β).
  - α and β were fitted on the training session's rest periods, which were themselves found from EMG.
  - So it is only partly independent of the new recipe.
- It scores 0.925 on the shared key, the same as the new pipeline (0.925). The old pipeline scores 0.540.

**The old pipeline graded against its own key (its most favourable scoring):** 0.683.

**The old key is physically implausible in places.** While all 4 muscles were silent, it gave:
- −2 Nm of "voluntary" torque for YES at −0.49 rad;
- −13 Nm for JM at −0.08 rad.

**Known built-in advantage:** phantom torque (mean |prediction| during rest) is partly zero by construction for the new pipeline.

**Results on the shared key:**

| | Average R² (8 angles) | R² at 2 most toes-down angles | RMSE at those angles |
|---|---|---|---|
| Main 3 (YES, JM, EG), old → new | 0.540 → 0.925 | 0.04 → 0.84 | 1.27 → 0.45 Nm |
| Extra 3 (IES, CH, HM), old → new | 0.874 → 0.966 | 0.70 → 0.97 | — |

- Main 3 use the average of 3 training runs; the extra 3 use 1 run.
- New beats old in all 12 person × direction comparisons.
- 47 of 96 per-angle results have R² ≥ 0.98, with errors of 0.3–3.0 Nm. These are mostly the strong angles, where the torque swings 15–30 Nm.

## 7. Is it data leakage?
Leakage means information from the test session reaching training or model choice.

**Checked, and not leakage:**
- **Training labels** use only the training session's torque. The test session's torque is used only to build the answer key for scoring, never as an input or a label. Every evaluation needs that.
- **The model's inputs are only EMG and angle.** B(t) is never an input.
- **Early stopping** uses the last 20 % of the training session, never the test session.
- **EMG scaling** uses the training-session maximum. Auto-gain does read the test session's EMG, but causally (past only) and with no torque, so it could run live.
- **Angle-sensor leak test:** replacing the measured angle (the footplate bends slightly during pushes) with the fixed set angle changed nothing (e.g. YES +0.03 rad: 0.99 → 0.99).
- **Code review:** an independent review for leakage and wrong metrics found 16 issues before the final runs, and all were fixed.

**Real leakage-like issues (acknowledged):**
- **Model selection:** we compared about 23 configurations on the test-session scores of the main 3, which inflates the winner. Mitigation: cross-fitting (choose the configuration on one direction, score it on the other: 0.932) and leave-one-subject-out (0.925).
- **The final choice was made after seeing the 3 extra people,** so they are not an independent confirmation. They had also been analysed in our earlier report.
- **The EMG smoothing is zero-phase,** so it uses about 0.25 s of future EMG. This applies equally to the old and new pipelines and is not label leakage, but it means the setup is offline only.
- **The shared definition** (EMG-quiet = zero voluntary torque) appears in both the new training labels and the answer key. This is "target definition coupling" rather than leakage, because the model never sees the key. It does inflate the new pipeline's phantom-torque score, and possibly its R² near rest.

**Not leakage, but optimistic:** the two sessions were in the same visit with the electrodes left on, so this is easier than testing on a different day.

## 8. Not done yet / limitations
- **Sensitivity:** we have not tested whether the conclusions survive other rest thresholds or smoothing. This needs no retraining, only re-scoring the saved predictions.
- **No independent ground truth** for voluntary torque (no ultrasound, nerve block or stimulation).
- **Low tone counted as passive:** steady muscle tone below the rest threshold is absorbed into the passive estimate.
- **Few rest moments in some trials**, e.g. JM session 2 at −0.50 rad is at rest only 0.9 % of the time.
- **Passive torque during contraction** may differ from passive torque at rest (MacIntosh 2005; Hoffman 2012).
- **Scope:** only 6 people, and both sessions were in the same visit.

## 9. Questions for the reviewer
1. Is defining voluntary torque through EMG-detected rest inside the same trial acceptable for evaluation? Is there precedent?
2. Which checks would convince you the improvement is not circular? Proposed checks:
   - (a) re-score with other rest thresholds (2 %, 5 %) and smoothing (σ = 4 s, 16 s);
   - (b) an offset-free metric, e.g. push size relative to the rest just before each push, which depends much less on the passive estimate;
   - (c) also report every model against the old key.
3. Should the weight of the foot and footplate be corrected separately? It is currently part of "passive".
