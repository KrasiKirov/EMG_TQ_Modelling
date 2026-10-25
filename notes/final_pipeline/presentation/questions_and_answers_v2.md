# Questions the supervisor may ask, with answers (deck v2)

Deck: `EMG_torque_supervisor_v2.pptx`. "S#" = slide number. Slides 45–50 are backup slides; keep them after the talk.

All numbers are graded with the shared answer key. "Main 3" = YES, JM and EG (3 training runs each). "Extra 3" = IES, CH and HM (1 run each).

## The experiment

**Did people only push, or also pull?** (S2)
- Five people only pushed the toes down.
- CH pulled the toes up at its most toes-down angles.
- At one angle (−0.31 / −0.33 rad) CH pulled in session 1 and pushed in session 2.
- The trial comments never say "pull", so ask the lab whether pulling was an instruction.

**Are session 1 and session 2 on different days?** (S2, S41)
- No. They were in the same visit, with the electrodes left on.
- A different-day test is still missing.

**What does the sensor measure?** (S5–S7)
- Measured torque = passive torque + voluntary torque.
- Passive torque is what the sensor reads when the muscles are off: the stretched tissues pulling back, plus the weight of the foot and footplate (Anderson 2010).
- We subtract the whole relaxed level measured in the same trial, so the weight of the foot and footplate is removed too.

**Why are the toes-down angles hard?** (S2, S7)
- With the toes down the calf is short and weak: people manage only 17–30 % of their strongest push.
- At YES −0.49 rad the pushes are about 1 Nm. Any error in the passive value is big compared with that.

## Passive torque

**What was the old passive table, and what was it used for?** (S9)
- It was one passive value per angle.
- It turned measured torque into the "right answer" for training and for grading: voluntary = measured − table value.

**How was the table computed?** (S9)
1. Before each pushing trial there is a separate relaxed recording at the same angle. Its first 20 s are a clean relaxed hold; after that the signals are unusable.
2. The code cut each recording where the angle signal jumps.
3. It kept every piece with a torque spread under 20 Nm and an average within ±20 Nm, and averaged each piece into one point.
4. It merged points closer than 0.05 rad and drew straight lines between them.
5. For each pushing trial it subtracted one number, read off that line at the trial's angle.

Exceptions:
- YES: the values were typed in from the lab worksheet.
- IES: there are no relaxed recordings, so nothing was subtracted.

**How did we see it was wrong?** (S10, S11)
- The JM bug:
  - Pieces that were not relaxed (torque jumping by ~40 Nm) passed the 20 Nm rule.
  - They made a +11.7 Nm table entry, so at −0.08 rad JM's "voluntary torque" sat about −13 Nm below zero while JM was relaxed.
  - Last semester's score hid it (R² 0.45), because it was graded against the same wrong answer. Against the shared answer key it is −60.
  - We traced it back to this table entry.
- Comparison with the trials:
  - We compared every table value with the relaxed level measured inside the pushing trial.
  - Even clean holds are off: the in-trial relaxed torque is about 80 % of the separate hold (median difference 0.6 Nm).
  - At the toes-down angles the table was too high in 20 of 20 trials, by up to 2 Nm.

**Why is the relaxed level inside the trial different from the separate recording?** (S11, S12)
- Stress relaxation: tissue held stretched slowly relaxes. The force drops by up to ~20 % within 60 s (McNair 2001) and keeps dropping for minutes (Duong 2001).
- Contractions make passive torque lower for a while, about 5 % (Kay & Blazevich 2009).
- Passive measures also differ between sessions (Hoang 2005; Walker 2022).

**What is passive drift and how did we find it?** (S12)
- The relaxed torque slides during a 90 s trial.
- We saw it by plotting the median torque of each relaxed stretch over time.
- It is usually small: median 0.15 Nm, and in 72 % of trials it moves toward zero.
- At YES −0.49 rad it is 0.5 Nm, against pushes of about 1 Nm.

**What does "measure it inside the trial" mean, concretely?** (S13)
1. The EMG tells us when all 4 muscles are switched off: average activity below 3 % of each muscle's range for at least 0.5 s.
2. In those moments the sensor reads only passive torque, so we take the median torque of each relaxed stretch.
3. We draw a smooth line through those medians (about 8 s smoothing).
4. Voluntary torque = measured − line.

**Isn't it circular to use EMG to find the relaxed moments?** (S13, S24)
- The EMG only says *when* the person is relaxed. The *value* comes from the measured torque.
- The line is smoothed over about 8 s, so it cannot copy pushes that last 1–2 s.
- The model never sees the line.

## EMG size and auto-gain

**Why would the EMG size change if the electrodes stayed on?** (S14)
- Electrode–skin contact changes over time, for example with sweat (Hewson 2003; Abdoli-Eramaki 2012).
- Skin temperature changes it (Winkel & Jørgensen 1991).
- Fatigue can change it either way (Dimitrova & Dimitrov 2003).
- People can share the push differently between muscles: at EG −0.74 rad, GM was 3.4× bigger while SOL was 0.45×.
- From our data we cannot tell which of these it was.

**How does auto-gain work, and is it published?** (S15)
- For each muscle, every 0.5 s:
  - floor = its quiet level over the past 20 s (10th percentile);
  - top = its strong-push level over the past 45 s (95th percentile);
  - output = (signal − floor) ÷ (top − floor).
- It uses only past EMG, never torque.
- It is switched off for people who push both ways (only CH).
- The idea is automatic gain control, as in radios and hearing aids.
- Running or sliding rescaling of EMG has been published (Koller 2015; Tanaka 2022), and using high percentiles is common (Jonsson 1982).
- Our exact running-percentile version is our own design.

**Why not divide by the MVC?** (S46)
- It scored worse (0.887; 0.904 per angle).
- The MVC was recorded once at the start of the visit, but the EMG size changes later.

## The fit and the model

**What does "divide by the typical swing" mean?** (S16, S17)
- σ = the standard deviation of the voluntary torque at that angle in the training data.
- We divide each angle's target by σ before fitting and multiply the prediction back by σ afterwards.
- This is weighted least squares with weight 1/σ² per angle, so every angle gets the same say. Without it, YES's two most toes-down angles carry 1 % of the total.

**Why 16 memory cells?** (S19)
- It was not a full search.
- Other sizes and common network designs were not better everywhere (our first report).
- 16 cells plus the per-angle corrections gave a small, consistent gain (+0.008) and fewer overshoot spikes.
- The model is still tiny: 3,665 numbers against about 57,000 training windows.

**What are the "experts"? Are they mini networks?** (S20)
- No. Each one is one extra set of 16 multipliers per angle, added to the shared read-out.
- They start at 0 and are kept small.
- Between two trained angles the two sets are blended.
- In fitting terms it is a shared slope plus a per-angle deviation pulled toward zero, like random slopes in a mixed-effects model.

**What does each change contribute?** (S21)
- We removed one change at a time from the new pipeline, with 1 training run each.
- See the table on S21.

## Checking

**How is the model tested? Can it overfit?** (S3, S23)
- We fit on the first 80 % of each training trial and use the last 20 % only to decide when to stop.
- We then predict the whole other session, which the model never saw, in both directions.

**What is the shared answer key, and is it biased toward the new method?** (S24, S41)
- It is measured − passive level measured inside that test trial, and it is used for every model.
- The version from our first report, whose answers are built differently, scores the same on it (0.925 vs 0.925).
- It does assume that quiet EMG means no push.

**What is phantom torque?** (S25)
- The torque the model predicts while the person is relaxed; it should be 0.
- Old: 1.25 Nm (main 3) and 1.07 Nm (extra 3). New: 0.17 Nm.

**Why 3 runs?** (S26)
- Fitting starts from random numbers, so each run gives a slightly different model.
- We average the 3 runs and never pick the best one.
- Old runs could disagree badly; the new runs agree.

**How was the final version picked?** (S27)
- Before looking at the extra people, we wrote a rule: highest average R² on YES, JM and EG.
- The rule picked a 0.4 s window version that still used the separate relaxed recordings.
- That version failed for IES (no relaxed recordings).
- At CH's flipped angle both versions failed until auto-gain was switched off for CH (now an automatic rule).
- We recommend the in-trial version instead. That choice was made after seeing the extra people.
- The extra people were also already looked at in our first report this semester, and CH's flip shaped the auto-gain on/off rule, so they are not a fully fresh test.

## Results

**Did it improve for everyone?** (S28–S31, S49)
- Main 3:
  - average R²: 0.54 → 0.92;
  - toes-down R²: 0.04 → 0.84;
  - toes-down error: 1.27 → 0.45 Nm.
- Extra 3:
  - average R²: 0.87 → 0.97;
  - toes-down R²: 0.70 → 0.97.
- New beats old at 88 of 96 single-angle results. At the toes-down angles of the main 3: 12 of 12 (Wilcoxon p = 0.0005, but 12 results from only 3 people). The other 8 are small drops at middle or toes-up angles, never toes-down.

**Isn't R² = 0.99 suspicious?** (S32)
- R² compares the error with how much the torque moves. At the strong angles the torque swings by ±15–30 Nm, so an error of 1–2 Nm still gives 0.99.
- Last semester's pipeline also had a median of 0.975 there.
- On the pushing parts only, the average is still 0.89.
- Even a straight-line model gets up to 0.975 there.
- No leak: the tested session is never used for fitting, and the code was reviewed for this.
- Published isometric work reports errors of 4–5 % of max effort; ours is about 3 %. But published work also drops on a different day, and we have not tested that.

**What do the titles "learned from session 1, tested on session 2" mean?** (S39, S40)
- The model was fitted on session 1 and then predicted session 2, which it never saw.

**Where are the full-trial plots for the other direction?**
- `figures_v2/inspect_<person>_S2toS1_old.png` and `..._new.png`.

## What else we tried (plain English; also on S46–S47)

- **Longer window (0.4 s instead of 0.2 s):**
  - slightly better overall (0.932 vs 0.925), but only at the strong angles;
  - promising: next we combine it with the new pipeline at 0.3–0.4 s.
- **Passive value from the relaxed hold just before each trial:** tie (0.921). Not kept.
- **Passive value that slowly relaxes during the trial (fitted drift curve):** tie (0.921). Not kept.
- **EMG ÷ MVC instead of auto-gain:** worse (0.887; per angle 0.904). Not kept.
- **Extra weight on the 2 most toes-down angles:** ×3 tie, ×10 worse. Not kept.
- **Randomly changing the EMG size during fitting:** helps only without auto-gain. Not kept.
- **Fitting on the other 5 people first, then fine-tuning:** worse (0.885). Not kept.
- **Removing the relaxed periods from the fit:** same R², but phantom torque goes from 0.25 to 0.36 Nm (tested on the earlier version from our first report). Not kept.
- **Scaling each session by its own overall EMG level:** 0.775, and it needs the test session in advance. Not allowed.
- **Straight-line model (ridge):** 0.624, or 0.863 with auto-gain. Not kept.
- **Old small network inside the new pipeline:** 0.921 vs 0.930 (1 run each), with more phantom torque. Simpler fallback.
- **Tests from our first report:**
  - other common network designs (GRU, CNN, plain network, bigger LSTM): none better;
  - angle-sensor leak test: passed;
  - re-zeroing with session 2's own torque: rejected as cheating.

## Sources

The full list is on S42–S43. Every source was opened and checked by a second reviewer.
