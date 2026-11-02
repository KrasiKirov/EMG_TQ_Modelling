# The "dorsiflexion" R² drop (that was actually plantarflexion)

*What broke in my EMG → ankle torque model, what I tested, and what fixed it.*

- **Date:** 2026-10-04
- **Branch / worktree:** `claude/dorsiflexion-r2-drop-341137` (nothing committed yet)
- **Subjects:** YES and JM (the broken ones), plus IES, EG, CH and HM as a check (CH ended up shaping the design a bit, see 7.5)

**Contents:** [TL;DR](#tldr) · [1. Setup](#1-the-setup) · [2. Data processing](#2-how-the-data-is-processed-and-what-the-old-model-looks-like) · [3. The problem](#3-the-problem) · [4. What went wrong](#4-what-was-actually-going-wrong) · [5. What I tried](#5-what-i-tried) · [6. Chosen solution](#6-the-solution-i-chose) · [7. Results](#7-results) · [8. Old vs new](#8-old-pipeline-vs-new-pipeline) · [9. Limitations](#9-limitations) · [10. Next time](#10-what-id-do-next-time) · [11. Glossary](#11-glossary-and-where-things-live) · [Appendix](#appendix-all-the-screening-runs)

*Jargon is explained in the glossary at the end (section 11).*

---

## TL;DR

- **The problem:** my model guesses ankle torque from muscle signals (EMG). In the second session (the held-out test data) it was great at most ankle angles but *worse than just guessing the average* at a few (R² below 0; R² = 1 is perfect). Worst: YES at −0.486 rad (≈ −28°), R² = −1.43.
- **Plot twist:** I'd been calling those the "dorsiflexion" (toes-up) angles. They're actually the most **plantarflexed** (toes-down) ones, where the calf is short and weak.
- **Why it broke:** at those angles the torque barely moves (~0.4 Nm of wiggle vs ~15 Nm at the strong angles), so errors of a few tenths of a Nm are enough to tank R². I found 5 sources of errors that size (section 4):
  1. the "relaxed" torque level shifting between sessions,
  2. the relaxed torque slowly drifting inside a trial,
  3. one garbage point in JM's passive-torque lookup table,
  4. EMG coming out bigger or smaller in the second session,
  5. training that basically ignored the weak angles.
- **What fixed it:** a cleaner target (fixes 1 and 3), an automatic EMG "volume knob" (fix 4), training where every angle counts the same (fix 5), plus one small output per angle. Number 2 is still not fixed.
- **Results (old → final, R² on session 2):** YES −0.486: −1.43 → 0.20 · JM −0.644: 0.02 → 0.80 · JM −0.360: 0.17 → 0.93. Runs with negative R² across all 6 subjects: 6 → 0 (out of 144). The overall error went down for all 6 subjects (JM: 2.05 → 0.33 Nm²).
- **What I picked:** the 4-change "final pipeline". The smaller 3-change "minimal fix" is actually better at YES −0.486 (0.47 vs 0.20) but doesn't fix JM's EMG problem, so it's plan B.

> **Names I use:** *old pipeline* (= `legacy` in the result files, `baseline` in the sandbox). *Minimal fix* (= `P0_core_w`, 3 changes). *Final pipeline* (= `train.py --recommended`; it shows up as `P11` in the results, or `P12` for CH, where the auto-gain switches itself off).

---

## 1. The setup

### What the model does

The model looks at muscle activity (EMG) from 4 lower-leg muscles plus the ankle angle, and guesses the **active ankle torque**, meaning the torque the muscles are making right now.

- **GM, GL** = gastrocnemius medialis / lateralis (the two halves of the big calf muscle)
- **SOL** = soleus (the calf muscle underneath)
- **TA** = tibialis anterior (the shin muscle that pulls the toes up)

GM, GL and SOL push the toes down (plantarflexion). TA pulls them up (dorsiflexion).

### The experiment

- The foot is strapped to a footplate that is **locked at one angle**. The person pushes against it but the ankle doesn't move (an **isometric** contraction).
- Each **session** has **8 trials**, one per ankle angle. Each trial is ~90 s, recorded at **1000 Hz** (1000 samples per second).
- Every subject did two sessions (both stored in the same recording file):
  - **Session 1** (the lab files call it "test") = the training data.
  - **Session 2** (the lab files call it "retest") = the held-out test set. The model never sees it during training.
- **One model per subject.** No mixing people.

### What one sample looks like

One sample is a **200 ms window**: 20 time steps (one every 10 ms) × 5 numbers (4 EMG envelopes + the ankle angle). The answer the model has to give is **one number**, the active torque at the last step of that window. For YES that's 56,896 training windows, 14,224 validation windows and 71,128 test windows (Figure 3).

### How I score it

The number I care about is **R² per ankle angle on session 2**.

**Units and stats I use everywhere**

- **Nm** (newton-metre) = unit of torque, i.e. how hard the foot twists against the plate.
- **rad** = angle unit. 1 rad ≈ 57°, so −0.49 rad ≈ −28°, −0.79 rad ≈ −45° and +0.25 rad ≈ +14°.
- **std** (standard deviation) = how much a signal typically wiggles around its average. Small std = the torque barely moves. **Variance** = std².

**The scores**

- **R²** (R-squared) asks: "how much better is the model than just guessing the average torque of that trial?"
  - 1 = perfect.
  - 0 = no better than guessing the average.
  - **Negative = worse than guessing the average.** That's the bad zone.
- Formula, for the curious: R² = 1 − MSE / variance of the true torque at that angle.
- **MSE** (mean squared error) = the average of (prediction − truth)², in Nm².
- **RMSE** = √MSE, so it's back in Nm. You can read it as "the typical error size".

---

## 2. How the data is processed (and what the old model looks like)

### Step by step

The numbers here match the panel numbers in Figures 1 and 2.

1. **Raw recording.** 6 channels at 1000 Hz: ankle angle, torque, and the 4 raw EMG signals.
2. **EMG envelope.** Raw EMG is a noisy buzz. I turn it into a smooth "how active is this muscle" curve called an **envelope**:
   - (2a) remove the offset (demean), keep only 30-300 Hz (band-pass filter, where the muscle signal lives), then flip the negative values up (rectify = absolute value),
   - (2b) smooth with a 2 Hz low-pass filter so only slow changes survive.
3. **Normalise.** (3a) Divide each muscle's envelope by its biggest value in session 1, so all 4 muscles sit roughly between 0 and 1. (3b, final pipeline only: the auto-gain from cause 4.)
4. **Target.** The torque sensor measures *everything*, including the "rubber band" torque the ankle makes even when relaxed. That part is the **passive torque**: stretched tendons and tissues pulling back. It's measured in separate relaxed "passive trials" and stored as a lookup table (angle → passive torque), the **passive map**. The target is **active torque = measured torque − passive torque at that angle**. (In the code this happens after step 5, but it's done sample by sample, so the order doesn't matter.)
5. **Trim + downsample.** Drop the first 1 s of each trial (start-up junk), then go from 1000 Hz to 100 Hz (low-pass first, so the downsampling doesn't create fake wiggles, then keep every 10th sample).
6. **Windows.** Slide a 20-step (200 ms) window along the trial, 1 step (10 ms) at a time. Each window gets the active torque at its last step as the answer.
7. **Split.** Session 1: first 80% of each trial's windows → train, last 20% → validation. Session 2: everything → test.

| # | Stage | Shape (one YES trial, or totals for row 7) | Units |
|---|---|---|---|
| 1 | Raw recording | ≈ 90,000 × 6 | rad, Nm, raw EMG (mV) |
| 2 | Envelope | ≈ 90,000 × 4 envelopes | mV, always ≥ 0 |
| 3 | Normalise | ≈ 90,000 × 4 | 0 to ~1, no unit |
| 4 | Target | ≈ 90,000 × 1 (one target per sample) | Nm |
| 5 | Trim + downsample | ≈ 8,900 × 6 (4 EMG + angle + target) | same |
| 6 | Windows | 8,891 × 20 × 5 inputs → 8,891 targets | mixed / Nm |
| 7 | Split | YES: 56,896 / 14,224 / 71,128 windows | — |

![What the signal looks like after each processing step](figures/fig03a_signal_stages.png)

*Figure 1: Real YES data after each step: raw EMG → rectified → envelope → scaled 0-1 (plus the auto-gain version the final pipeline uses, 3b) → measured torque → target. The target shown is the old pipeline's, so its rest sits at about +1.1 Nm instead of exactly 0 (that's cause 1).*

![What one model input window looks like](figures/fig03b_model_input.png)

*Figure 2: Downsampling to 100 Hz (step 5) and cutting one 20 × 5 input window (step 6). The model reads the 20 rows in order and outputs one torque value.*

![How the data is split into train, validation and test](figures/fig04_data_split.png)

*Figure 3: The split (step 7). In each session-1 trial the first 80% is used for training and the last 20% for validation. The whole of session 2 is the test set.*

### The old model

The old model is a tiny 2-layer **LSTM**. An LSTM is a kind of neural network that reads a sequence one step at a time and keeps a small "memory", so it can use what happened earlier in the window.

- LSTM with 8 units → **dropout** 30% → LSTM with 8 units → dropout 30% → 1 output neuron (the torque).
- **Dropout** = during training, randomly switch off 30% of the neurons at each step. That stops the network from memorising the training data too hard. Memorising is called **overfitting**: great on training data, bad on new data.
- **1,001 trainable numbers** (parameters) in total. Really small.
- Training: MSE loss in Nm², Nadam optimiser (the rule that nudges the weights after each batch), learning rate 0.003 (how big each nudge is), **batch size** 8 (how many windows it looks at before each nudge), at most 200 epochs (full passes over the training data).
- **Validation** = the 20% of session-1 data the model doesn't train on. I use it to know when to stop. **Early stopping** ends training when the validation error hasn't improved for 25 epochs, then keeps the best version.

![Old model architecture](figures/fig05_arch_old.png)

*Figure 4: The old model. A 20 × 5 window goes through two small LSTM layers (8 units each, with dropout) and comes out as one torque value.*

---

## 3. The problem

Most angles were great (R² 0.93-0.99 for YES). But a few were terrible:

- **YES −0.486 rad:** R² **−1.43** (all 3 training runs negative: −1.10 / −1.65 / −1.55).
- **JM −0.787 rad:** **−0.81** on average. Honestly that one is noisy: two runs gave 0.61 and 0.89, the third gave −3.93.
- **JM −0.644 rad:** **0.02**. **JM −0.360 rad:** **0.17**.
- **JM −0.078 rad:** **−2.06** (that one turned out to be a bug, see cause 3).

(These are means of 3 runs with the faster batch-64 training I used for all the testing. With the original batch-8 settings (1 run, seed 0) it's the same story: YES −0.486 = −0.68 and JM −0.644 = −0.32. See 7.3.)

![Old pipeline R² per angle for YES and JM](figures/fig01_problem_r2.png)

*Figure 5: The old model's R² per angle in session 2. Below the dashed line = worse than guessing the average. The bad angles are mostly at the toes-down end (JM −0.078 is a separate bug, see cause 3). Where the 3 runs disagree wildly, the black dots show each run.*

### Wait, those angles are plantarflexed, not dorsiflexed

I had been calling these the "dorsiflexion" angles. That's backwards. Negative angle = **plantarflexion** (toes down). The evidence:

- **The lab code says so:** `maxDF = +0.25` and `maxPF = -0.50` rad (DF = dorsiflexion, PF = plantarflexion), in `EMG2TQ_TIDynNLBiLinModel_prbsTQ_HM01_20250411.m`. And `maxPF = -0.5`, `maxDF = +0.2` in `initialize_input_waveform.m`. So positive = dorsiflexion, negative = plantarflexion, and plantarflexion torque is negative.
- **Max push is weakest at the negative angles.** The **MVC** (maximum voluntary contraction, i.e. pushing as hard as you can) for YES is **13.6 Nm at −0.488 rad** vs **81.6 Nm at +0.139 rad**, only 17% of the peak. Across all 6 subjects the most negative angle gets just 17-30% of their peak. That's exactly what happens when the calf is already short (foot pointed down). It's like trying to curl a dumbbell when your arm is already fully bent.

![Ankle angle convention and MVC per angle](figures/fig02_ankle_angles.png)

*Figure 6: Left: which way the angles go (from the lab code: negative = toes down). Right: the hardest YES and JM could push at each angle. The "bad" angles are the toes-down ones, where the calf is weakest.*

- **The passive torque sign fits.** For YES the passive torque is **+7.9 / +10.6 Nm** (session 1 / session 2) at −0.487 rad and **−10.1 / −5.0 Nm** at +0.245 rad. Positive (pulling toes up) at negative angles means the front of the ankle is stretched, which happens when the foot points down. And the other way around at positive angles.
- **The range of motion fits.** The recorded angles go from about −0.79 rad (−45°) to +0.25 rad (+14°). Ankles go way further toes-down than toes-up, so a range that reaches −0.79 rad only makes sense if negative = plantarflexion.

Why this matters: the explanation flips. It's not "the calf is stretched and weird". It's "**the calf is short and weak, so there's hardly any torque to predict**". (`notes/YES_passive_torque_issue.md` in the main project folder calls −0.487 "maximum dorsiflexion", which is wrong for the same reason.)

### Why R² is so harsh here

R² compares your error to how much the torque moves. At the weak angles the torque barely moves at all:

| YES angle | Torque std in session 2 | Old RMSE | Old R² |
|---:|---:|---:|---:|
| −0.486 | 0.40 Nm | 0.62 Nm | −1.43 |
| +0.138 | 14.95 Nm | 2.53 Nm | 0.97 |

So at +0.138 the typical error is **4× bigger** in Nm, and R² still looks amazing. At −0.486 a typical error of 0.62 Nm (tiny!) gives R² −1.43.

It's like grading someone's guess of a 1 cm plant's height: being off by 6 mm is a terrible guess, while being off by 2.5 cm on a 15 m tree is basically perfect. R² grades you relative to how much there is to predict. The annoying part is that it makes a small absolute problem look like a disaster. That's why I report RMSE next to R² from now on.

---

## 4. What was actually going wrong

Since the torque at the weak angles only moves ~0.4-0.8 Nm (std), anything that adds a few tenths of a Nm of error kills R² there. I found 5 such things.

(A few numbers in this section, like trial 21, the 0.33 slope, YES's ~1.3×, the 0.53 Nm slide, the ~1.2 Nm push, the ~6× electrode pop and the 10 → 20 Nm filter change, come from my earlier notes, `notes/dorsiflexion_investigation.md`, not from the final result files.)

### Cause 1: the passive offset changes between sessions

- The target is "measured torque − passive torque". The passive torque comes from separate relaxed holds.
- But during the *active* trials, the relaxed torque between pushes is **not** the same as in the passive holds. For YES it's about **0.78 × the passive-hold torque + 0.30 Nm** (fitted on session 1). My guess is the tissue gets a bit "looser" after being contracted a few times (the fancy word is thixotropy).
- On top of that, the passive value itself was different in the two sessions at YES −0.487: **7.9 Nm in session 1** (that one is the lab worksheet value, there's no clean static hold at −0.49 in that session) vs **10.6 Nm in session 2**.
- Result: plain subtraction leaves a different leftover offset in each session. The rest level of the old target at YES −0.49 was **−1.32 Nm in session 1 and −1.98 Nm in session 2**. That's a **0.65 Nm jump**, while the whole signal there has a std of only **0.40 Nm** (session 2).
- Quick math: if the prediction is off by a constant 0.65 Nm and the torque only wiggles by 0.40 Nm (std), then R² = 1 − (0.65/0.40)² ≈ **−1.6**. The offset alone explains the whole disaster.

**Fix: passive correction.** Use P_eff = α·P + β instead of P, with α and β fitted on session 1 only. To fit them I use the rest periods, which are found from EMG (muscles silent), never from torque.

**Why fitting on session 1 also fixes session 2:** each session already gets its own passive map (that's why it's 7.9 in session 1 and 10.6 in session 2). The real problem is that during the pushing trials the ankle only "feels" ~78% of that passive torque. Plain subtraction over-subtracts by ~22% of P, so the bigger P is, the bigger the leftover. P was bigger in session 2, so the leftover was bigger there. Check: −0.22 × 7.9 + 0.30 ≈ −1.4 Nm and −0.22 × 10.6 + 0.30 ≈ −2.0 Nm, close to the measured −1.32 and −1.98. α and β capture that "78% plus a bit" relationship, which seems to stay the same between sessions even when P changes. After the fix the two sessions rest within 0.05 Nm of each other (Figure 7).

Fitted values: α = 0.78 (YES), 1.03 (JM), 0.83 (EG), 0.78 (CH), 0.82 (HM). For JM α ≈ 1, so the "looser tissue" effect isn't there for everyone. IES has no passive map at all (none was found in either session), so there's nothing to fit.

![Passive offset between sessions at YES −0.49 rad](figures/fig08_passive_offset.png)

*Figure 7: YES at −0.49 rad. Top: with the old target, the resting level sits 0.65 Nm lower in session 2, which is bigger than the std of the whole signal there (0.40 Nm). Bottom: with the passive correction both sessions rest at the same level (0.05 Nm apart). The slow slide inside the trial is cause 2.*

### Cause 2: passive torque relaxes during a trial

- Even within one 90 s trial, the relaxed level slowly slides down (the tissue relaxes under constant stretch).
- At YES −0.486 the fitted rest-level slope was about **0.53 Nm over 90 s**, while the whole push there (rest to peak) is only **~1.2 Nm**. At every other YES angle it's negligible.
- I tried cleaning this out of the training labels (`H_pgain_smooth_posw`), and honestly it didn't help on session 2 (YES −0.486: 0.33 vs 0.38 without it).
- The only things that really remove it peek at the session-2 torque itself (re-zeroing each session-2 trial, see the appendix, table E), which is cheating. So this one is **not fixed**, and it's a big part of why YES −0.486 is still the weakest angle. The real fix is in the recording protocol (section 10).

### Cause 3: a bug in JM's passive map

- The passive map gets built automatically from the relaxed trials, with a sanity filter that throws out crazy torque values. At some point that limit got bumped from 10 to 20 Nm (for YES, even though YES ends up using hand-entered worksheet values anyway).
- With the looser limit, a bit of JM's trial 21 where the foot was actually *moving* got counted as a relaxed hold, in both sessions: **+11.72 Nm at −0.071 rad** (session 1) and **+10.75 Nm at −0.064 rad** (session 2). The neighbours are −0.51 Nm (at −0.217) and −3.38 Nm (at +0.059), so it's obviously garbage (Figure 8).
- Effect: at JM −0.078 the old target sat at **−13.6 Nm (session 1) / −12.2 Nm (session 2)** even with JM fully relaxed, while the torque there only wiggles by 1.63 Nm (std). The two sessions also differ by 1.4 Nm. Old R²: **−2.06**, and one of the 3 runs even gave −7.23.

**Fix: static-hold passive map.** Every passive trial starts with a 20-30 s stretch where the foot is completely still. I only use that part (torque std below 1 Nm, foot wobble below 5 mrad ≈ 0.3°) and ignore the rest.

- With the final pipeline JM −0.078 goes from R² −2.06 → **0.87** (MSE 8.13 → 0.37 Nm²), and the minimal fix already gets 0.59 (1.15 Nm²).
- In the ridge model, where this is the only change (ridge A → B), JM −0.078 goes 0.25 → 0.87.
- That one angle was about half of JM's old pooled MSE (8.13 out of 8 × 2.05 ≈ 16.4), and about 56% of the drop from 2.05 to 0.33. That's why JM's overall number drops so much.

![JM passive map bug](figures/fig09_jm_passive_bug.png)

*Figure 8: JM's passive map. The old auto-detected map (blue) has one bogus point near −0.07 rad from a bit of a movement trial. The new static-hold map (orange) is smooth. Everywhere else the two maps are the same.*

### Cause 4: EMG amplitude drifts between sessions

- Something about the EMG changed between the two sessions: electrode contact, sweat, fatigue, maybe the electrodes were moved (honestly I don't know which). So **the same push can give a different EMG size in session 2**.
- For JM, the calf EMG swing in session 2 was only **0.36×** (at −0.64 rad), **0.43×** (at −0.36) and **0.69×** (at −0.79) of session 1, for about the same torque.
- So the model under-predicts: at JM −0.644 the old model's predictions only had about a third of the real amplitude (slope 0.33). YES went the other way at −0.381 (EMG ~1.3× bigger, predictions too big).

**Fix: causal EMG auto-gain**, basically an automatic volume knob per muscle:

- floor = 10th percentile of the last 20 s (≈ what this muscle looks like at rest; the 10th percentile is the value that 10% of the samples are below),
- top = 95th percentile of the last 45 s (≈ what a real push looks like),
- new value = (envelope − floor) / (top − floor), clipped to 0-2.

**Why it works:** if the electrodes give half the signal in session 2, the floor and the top also halve, so the ratio comes out the same. The model sees "pushing at your usual level" in both sessions (Figure 9).

It only uses the **past** ("causal"), so it could run live. At the very start of a trial, with no past yet, it uses numbers from session 1. It holds the last good value through artefact spikes.

Evidence it's the key for JM: in my quick-test setup (the "sandbox", see section 5), auto-gain + angle weights with no passive fix takes JM −0.644 from −0.04 to **0.79** and JM −0.360 from 0.41 to **0.91**. Even a plain linear model gets much better with it (ridge with the passive fixes, YES pooled MSE: 17.05 → 6.38 once auto-gain is added).

**The catch: CH.**

- CH pushes **both ways** in session 1 (per-trial push values from −54.8 Nm toes-down to +28.0 Nm toes-up).
- With auto-gain forced on, CH's pooled MSE blows up to **78.70 Nm²** (vs **3.58** without).
- My guess: when a muscle is quiet for a whole trial, auto-gain still boosts its noise (up to ~3×, there's a cap), and with pushes in both directions the model can't tell real activity from boosted noise.
- So the final pipeline has an `auto` mode: it checks the push directions in session 1 and turns auto-gain off if it sees both. Only CH triggers it, out of the 6.

![EMG gain drift between sessions](figures/fig10_emg_gain_drift.png)

*Figure 9: JM at −0.36 rad. The torque swing is about the same in both sessions, but the summed calf EMG swing is 2.3× smaller in session 2. After the causal auto-gain the swings line up again.*

### Cause 5: the loss basically ignores the weak angles

- The old loss is plain MSE in Nm². Being 1 Nm off costs the same everywhere, but 1 Nm is nothing at an angle where the torque swings 15 Nm and huge where it swings 0.7 Nm.
- In YES's training data, −0.49 rad is only **0.06%** of the total loss, while +0.14 rad is **30.6%** (Figure 10). In std terms that's a gap of about 480× (std 0.69 vs 15.2 Nm, squared). On session 2 it would be even bigger, about 1400× ((14.95 / 0.40)²). The model basically never gets told off for being wrong at the weak angle.
- Angle weighting alone doesn't fix anything (YES −0.486: −1.16, still negative), because the target itself was wrong. Once the target is fixed you need it though: without it the model kept a small resting offset at the weak angles (about 0.5 Nm in my earlier notes).

**Fix:** measure each angle's error in units of *its own* wiggle. The minimal fix weights each angle's error by 1/std². The final pipeline divides each angle's target by its std before training and multiplies the prediction back after. Same idea both ways: being off by "one wiggle" costs the same at every angle, so each of the 8 angles gets 12.5% of the loss.

![Loss balance across angles](figures/fig11_loss_balance.png)

*Figure 10: YES training data (session 1). Left: how much the torque wiggles at each angle. Right: each angle's share of the training loss. With plain MSE the most plantarflexed angles count for almost nothing (−0.49: 0.06%, −0.38: about 1%); with per-angle scaling every angle gets 12.5%.*

### Things I suspected that turned out NOT to be the cause

- **Angle sensor "leak".**
  - *Worry:* the footplate bends a tiny bit when you push, so the measured angle wiggles along with the torque. At the strong YES angles that wiggle alone explains most of the torque (straight-line fit R² 0.98 at +0.031, 0.97 at −0.071, 0.93 at +0.138; only 0.02 at −0.486). A model with *only* the angle, no EMG, even gets 0.64 at −0.071 and 0.75 at +0.031 (but only 0.09-0.44 at the other strong angles). So maybe the LSTM was cheating through the angle.
  - *Test:* swap the measured angle for the fixed angle the plate was set to.
  - *Result:* no change (YES +0.031: 0.99 → 0.99, −0.071: 0.98 → 0.98, +0.138: 0.97 → 0.97; same for HM and EG). The LSTM isn't using the wiggle. Not the cause.
- **Electrode artefact spike.** JM −0.360 in session 2 has a big electrode "pop" on GL (~6× the training max). Clipping it only moved R² from 0.41 to 0.49. Real, but small. The auto-gain handles spikes anyway.
- **"Maybe the model is just too small / the wrong type."** On the same fixed data, a GRU, a bigger LSTM, a 1D-CNN and a plain MLP don't do better at the weak angles, and some do much worse (section 5, Figure 11). The architecture wasn't the bottleneck. The data was.

---

## 5. What I tried

I screened **about 40 variants**. Each one was trained with **3 random seeds** and I report the **mean**, with no seed picking. A **seed** is the random number that decides the starting weights and the shuffle order, so different seeds give slightly different models.

They ran in two places:

- **"sandbox"** = my quick experiment setup (`ML/experiments/harness.py`), built for fast screening.
- **"real code"** = the actual training pipeline (`ML/training/train.py`).

The two don't give *identical* numbers for the same setup (e.g. old pipeline at YES −0.486: −1.09 in the sandbox vs −1.43 in the real code), so I only compare rows from the same place. All runs use a **"fast" protocol**: batch 64 instead of 8, at most 150 epochs instead of 200, stop after 15 epochs without improvement instead of 25, and the learning rate gets lowered faster. Otherwise screening would have taken forever. Numbers = R² on session 2 at that angle.

**Words used in the tables:** *passive fixes* = static-hold map + passive correction (causes 1 and 3) · *weights* = angle-weighted loss (cause 5) · *auto-gain* = the EMG volume knob (cause 4) · *dual view* = feed both the normal and the auto-gained EMG · *MoE* (mixture of experts) = one small expert output per angle (section 6) · *mean prior* = each expert also learns "the usual torque at this angle" · *scale-only* = no such offset · *drift-free labels* = training targets with the within-trial passive slide removed (cause 2) · *set-point angle* = the angle the plate was set to · *zero-anchored* = output forced to 0 when the EMG is silent · *fair* = no session-2 torque used anywhere (see the appendix).

### Short version

- **If you don't fix the target, YES −0.486 stays negative**, whatever else you do: clipping (−1.30), set-point angle (−0.76), angle weights alone (−1.16), MoE alone (−1.09), all sandbox. The problem was the target, not the model.
- **Fixing the passive part flips YES −0.486 positive:** 0.47 with the minimal fix, 0.46-0.53 with passive fixes + MoE (P12 0.46, H_pgain_moe 0.51, P5 0.53).
- **JM needs auto-gain.** Without it, JM −0.644 never gets above **0.37** in any fair variant (real code, sandbox or ridge). With it: **0.60-0.85**.
- **Auto-gain costs YES −0.486 a bit.** In the real code, LSTM variants with auto-gain land at 0.15-0.28 there, vs 0.46-0.53 without. That's the main trade-off of the final pipeline (my guess why is in section 6).
- **The model family doesn't fix it.** On the same fixed data no family beats the LSTM 8-8 everywhere, and some fall apart on certain angles (EG −0.739: MLP −0.20, LSTM 32-32 0.05 vs LSTM 8-8 0.79; JM −0.787: GRU 0.34 vs 0.87). See Figure 11. A plain linear model does surprisingly OK on JM (small torques), but it's way worse on subjects with big torques (original-data ridge, YES pooled MSE: 16.72 vs 2.36 for the old LSTM). It also improves a lot once auto-gain is added.
- **The cheating variants don't win overall.** They're somewhat better at YES −0.486 (best 0.66 vs 0.53 fair), but clearly worse at JM −0.644 (best 0.49 vs 0.85 fair). So touching the session-2 torque isn't worth it.
- **Why "scale-only":** the "mean prior" MoE (P5, P8) looked fine for YES and JM, but it learns "at this angle the torque is usually around X". CH did a **dorsiflexion** push at −0.33 rad in session 1 (TA, mean +13.4 Nm) and a **plantarflexion** push in session 2 (GM/GL, mean −15.6 Nm), so a mean prior is just wrong there. That's why the final uses scale-only.

### The main table (real code only, so every row is comparable)

| Setup (real code) | What's in it | YES −0.486 | YES −0.381 | JM −0.644 | JM −0.360 |
|---|---|---:|---:|---:|---:|
| old pipeline (`legacy`) | nothing changed | −1.43 | 0.76 | 0.02 | 0.17 |
| minimal fix (`P0_core_w`) | passive fixes + weights, same LSTM 8-8 | 0.47 | 0.75 | 0.24 | 0.64 |
| `P5_core_moe` | passive fixes + MoE with mean prior | 0.53 | 0.85 | 0.18 | 0.59 |
| `P12_core_moe_scale` (final for CH) | passive fixes + MoE scale-only | 0.46 | 0.83 | 0.22 | 0.66 |
| `P7_core_agc_w` | passive fixes + auto-gain + weights | 0.24 | 0.86 | 0.73 | 0.92 |
| `P10_core_dual_w` | passive fixes + dual view + weights | 0.28 | 0.72 | — | — |
| `P8_core_agc_moe` | passive fixes + auto-gain + MoE with mean prior | 0.15 | 0.88 | 0.85 | 0.94 |
| `P9_core_dual_moe` | passive fixes + dual view + MoE | 0.22 | 0.86 | 0.60 | 0.48 |
| **final (`P11_core_agc_moe_scale`)** | passive fixes + auto-gain + MoE scale-only | 0.20 | 0.87 | 0.80 | 0.93 |

"—" = not run for that subject. P8 is a hair better than the final on JM, but its mean prior breaks for CH (see "Why scale-only" above). All the sandbox runs (about 30 more, including the cheating ones) are in the [appendix](#appendix-all-the-screening-runs).

![Other model families on the same fixed data](figures/fig16_model_families.png)

*Figure 11: MLP, 1D-CNN, GRU, bigger LSTM, my original LSTM 8-8 and a straight-line ridge model, all trained on the same minimal-fix data (sandbox runs), at the two weakest angles of YES, JM and EG. No model wins everywhere, so the model type wasn't the problem. (The LSTM 8-8 here is the sandbox minimal fix, `H_pgain_posw`; the real-code minimal fix gets 0.47 at YES −0.486.)*

---

## 6. The solution I chose

**Final pipeline = 4 changes**, run with:

```bash
python ML/training/train.py --subject YES --recommended
```

![Old vs new pipeline](figures/fig07_pipeline_old_vs_new.png)

*Figure 12: Old pipeline vs new pipeline, step by step. The changed steps are the passive map, the passive correction, the EMG auto-gain and the model / loss.*

### The 4 changes

1. **Static-hold passive map.** Build the passive map only from the clean still holds at the start of each passive trial. That stops movement junk from sneaking in, and it's what fixes the JM bug. (YES uses hand-written worksheet values in both pipelines, so this change only matters for the other subjects, mainly JM.)
2. **Passive correction.** Subtract α·P + β instead of P, with α and β fitted on session 1's rest periods (found from EMG, not torque). This removes most of the session-to-session offset. α and β are fitted without any session-2 data. The session-2 target still uses session 2's own passive-hold map, same as the old pipeline.
3. **Causal EMG auto-gain** (`auto` mode). Each muscle's envelope is rescaled by its own recent range, using only the past. That takes care of the EMG coming out a different size in session 2. It switches itself off when session 1 has pushes in both directions (CH).
4. **A slightly bigger LSTM with one small "expert" per angle, trained so every angle counts the same.**
   - Shared part: same kind of LSTM as before, just wider (16-16 instead of 8-8, dropout 20%).
   - On top: one shared output, plus one tiny linear "expert" for each of the 8 training angles. The model reads the ankle angle and adds that angle's expert correction (between two angles it blends both).
   - Why: the EMG → torque relationship changes with muscle length, so each angle gets its own small fine-tuning knob on top of one shared brain.
   - The experts start at zero and get a penalty for growing big (an "L2 penalty"), so they stay small corrections.
   - Target: each angle's torque divided by its own std (cause 5). This replaces the angle weights from the minimal fix (same idea, different way).
   - No per-angle "usual torque" offset, because CH pushed the opposite way at −0.33 rad in session 2.
   - 3,665 parameters, still tiny.

![New model architecture](figures/fig06_arch_new.png)

*Figure 13: The new model. A shared LSTM 16-16 part feeds a shared output plus one small linear expert per ankle angle. The model reads the ankle angle and picks that angle's expert (between two training angles it blends their two experts), and the output is scaled back to Nm with that angle's std.*

### The simpler fallback: the minimal fix

If I want something small and easy to explain, the **minimal fix** is changes 1 + 2 + angle-weighted loss, with the **same old LSTM 8-8**.

- About **100-150 lines** of code (about 60 without comments).
- Removes **every** negative R²: 0 out of 144 single runs across 6 subjects.
- **Better than the final at YES −0.486: 0.47 vs 0.20** (RMSE 0.29 vs 0.35 Nm).
- But it barely helps JM's EMG problem (JM −0.644 only 0.24, −0.360 0.64), and pooled MSE doesn't really improve for YES (2.43 vs 2.36 old), CH or HM.

**Why does auto-gain hurt YES −0.486?** I don't know for sure, I haven't tested it. My guess: at that angle the calf barely switches on, so the "top" of the recent range is small and noisy, and dividing by it boosts the noise, a bit like the CH problem.

That's why I went with the 4-change version as the main one, and keep the minimal fix as plan B.

### Code size (being honest)

| What | Lines |
|---|---:|
| Minimal fix | ~100-150 (about 60 without comments) |
| Everything added to the real pipeline (both fixes, auto-gain, MoE, flags) | 770 |
| Experiment scaffolding (sandbox, plugins, job queue, reports), not needed to use the pipeline | 3,133 |

---

## 7. Results

All numbers below: real code, session 2, mean of 3 seeds, fast protocol (batch 64, max 150 epochs), unless I say otherwise.

![YES and JM results old vs minimal vs final](figures/fig12_results_yes_jm.png)

*Figure 14: R² per angle for YES and JM: old pipeline, minimal fix and final pipeline. After the fixes no angle is below zero anymore.*

### 7.1 YES, angle by angle

| Angle (rad) | Torque wiggle in session 2, std (Nm) | R²: old → minimal → final | RMSE (Nm): old → minimal → final |
|---:|---:|---|---|
| −0.486 (≈ −28°) | 0.40 | −1.43 → 0.47 → 0.20 | 0.62 → 0.29 → 0.35 |
| −0.381 (≈ −22°) | 2.24 | 0.76 → 0.75 → 0.87 | 1.08 → 1.12 → 0.80 |
| −0.278 (≈ −16°) | 4.60 | 0.93 → 0.94 → 0.99 | 1.25 → 1.10 → 0.47 |
| −0.176 (≈ −10°) | 7.49 | 0.98 → 0.98 → 0.99 | 1.12 → 1.10 → 0.85 |
| −0.071 (≈ −4°) | 8.56 | 0.98 → 0.98 → 0.99 | 1.13 → 1.21 → 0.80 |
| +0.031 (≈ +2°) | 13.63 | 0.99 → 0.99 → 0.99 | 1.42 → 1.64 → 1.60 |
| +0.138 (≈ +8°) | 14.95 | 0.97 → 0.96 → 0.98 | 2.53 → 2.81 → 2.12 |
| +0.245 (≈ +14°) | 12.96 | 0.97 → 0.98 → 0.98 | 2.17 → 1.84 → 1.77 |

**All angles together:** MSE 2.36 → 2.43 → 1.57 Nm², RMSE 1.53 → 1.56 → 1.25 Nm (old → minimal → final). The only angle where the final's error got slightly worse than the old one is +0.031 (MSE 2.03 → 2.55 Nm²), and R² there is still 0.99.

### 7.2 JM, angle by angle

| Angle (rad) | Torque wiggle in session 2, std (Nm) | R²: old → minimal → final | RMSE (Nm): old → minimal → final |
|---:|---:|---|---|
| −0.787 (≈ −45°) | 0.60 | −0.81 → 0.89 → 0.86 | 0.64 → 0.20 → 0.22 |
| −0.644 (≈ −37°) | 0.78 | 0.02 → 0.24 → 0.80 | 0.77 → 0.68 → 0.35 |
| −0.504 (≈ −29°) | 1.85 | 0.90 → 0.88 → 0.88 | 0.57 → 0.65 → 0.65 |
| −0.360 (≈ −21°) | 1.32 | 0.17 → 0.64 → 0.93 | 1.20 → 0.79 → 0.35 |
| −0.218 (≈ −12°) | 1.35 | 0.92 → 0.90 → 0.91 | 0.39 → 0.42 → 0.40 |
| −0.078 (≈ −4°) | 1.63 | −2.06 → 0.59 → 0.87 | 2.30 → 1.06 → 0.61 |
| +0.055 (≈ +3°) | 2.69 | 0.71 → 0.55 → 0.92 | 1.40 → 1.70 → 0.71 |
| +0.199 (≈ +11°) | 2.96 | 0.66 → 0.68 → 0.90 | 1.68 → 1.56 → 0.92 |

**All angles together:** MSE 2.05 → 1.08 → 0.33 Nm², RMSE 1.37 → 1.02 → 0.57 Nm (old → minimal → final).

JM's torques are small everywhere (std ≤ 3 Nm), so every angle is a "weak" angle for JM. That's why the gain is so big overall.

![Prediction traces old vs new](figures/fig14_prediction_traces.png)

*Figure 15: 50 s of one session-2 trial (seed 0). Black = the target (measured torque minus the passive estimate), colour = the model's guess. Heads-up: the black line isn't identical in the old and new panels, because the new passive correction shifts the target (by 2.05 Nm for YES, 0.30 Nm for JM). Old model: YES's guess sits ~0.8 Nm off at rest, and JM's guesses are way too small during pushes. New model: both follow the pushes. (Single-run R² in the plot differs a bit from the 3-run means in the tables.)*

### 7.3 Check with the original training settings (batch 8, seed 0)

Most numbers above use the fast protocol. To make sure the fix isn't just a side effect of that, I re-ran the old and final pipelines once with the **original** settings (batch 8, learning rate 0.003, at most 200 epochs, early-stopping patience 25), seed 0.

| YES angle | R² old → final | JM angle | R² old → final |
|---:|---|---:|---|
| −0.486 | −0.68 → 0.17 | −0.787 | 0.66 → 0.88 |
| −0.381 | 0.75 → 0.88 | −0.644 | −0.32 → 0.84 |
| −0.278 | 0.95 → 0.99 | −0.504 | 0.90 → 0.88 |
| −0.176 | 0.96 → 0.99 | −0.360 | 0.46 → 0.93 |
| −0.071 | 0.98 → 0.99 | −0.218 | 0.90 → 0.90 |
| +0.031 | 0.99 → 0.98 | −0.078 | 0.56 → 0.84 |
| +0.138 | 0.97 → 0.98 | +0.055 | 0.49 → 0.93 |
| +0.245 | 0.98 → 0.98 | +0.199 | 0.69 → 0.85 |

(The old RMSE wasn't saved for this re-run, so I only show R².) Same picture: the bad angles get fixed and the good ones stay good.

### 7.4 Stability across seeds

The final model is way more repeatable: the biggest seed-to-seed spread (std of R² over the 3 runs, at any angle) is 0.03 or less for every subject. Before, JM's result basically depended on which seed you got (−0.787: 0.61, 0.89, −3.93). The per-subject numbers (max seed std, worst single run) are in the section 8 table.

![Seed stability](figures/fig15_seed_stability.png)

*Figure 16: One dot per training run (seed) at the weak angles of YES and JM. The old pipeline's runs are all over the place (JM −0.787: 0.61, 0.89, −3.93). The final pipeline's runs sit on top of each other.*

### 7.5 All six subjects

![All subjects old vs final](figures/fig13_all_subjects.png)

*Figure 17: All 6 subjects, old vs final. Left: R² at the most plantarflexed angle. Right: overall error (pooled MSE). Everyone improves, including IES, EG and HM, which weren't used to pick the method (CH did shape the design, see below).*

- **IES, EG and HM weren't used to pick the method.** EG and HM only show up in side tests (the angle-leak check, the model-family runs), and IES wasn't touched at all until the end, so IES is the cleanest "fresh" check.
- **IES has no passive map at all**, so the two passive fixes don't do anything for IES. Its improvement (0.70 → 0.97 at −0.591) comes from the auto-gain and the per-angle model + scaling. That makes it a nice separate check of those parts.
- **CH did influence the design a bit:** its task flip is why the final is scale-only and why auto-gain has an `auto` switch.

---

## 8. Old pipeline vs new pipeline

### All the numbers, all subjects

The angle next to each subject is that subject's most plantarflexed (most negative) angle. "Worst-angle R²" = the lowest of the 8 angles. Mean R² is the plain average over the 8 angles. Pooled MSE is over all 8 angles together. "Worst single run" = the lowest single-run R² at any angle. "Max seed std" = the biggest seed-to-seed spread of R² at any angle. Negative runs = single-run R² values below 0, out of 8 angles × 3 seeds = 24.

| Subject (most plantarflexed angle) | Pipeline | R² at that angle | Worst-angle R² (angle) | Mean R² (8 angles) | Pooled MSE (Nm²) | Worst single run | Max seed std | Negative runs (of 24) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YES (−0.486) | old | −1.43 | −1.43 (−0.486) | 0.64 | 2.36 | −1.65 | 0.24 | 3 |
| YES (−0.486) | minimal | 0.47 | 0.47 (−0.486) | 0.88 | 2.43 | 0.40 | 0.08 | 0 |
| YES (−0.486) | **final (P11)** | 0.20 | 0.20 (−0.486) | 0.87 | 1.57 | 0.18 | 0.02 | 0 |
| JM (−0.787) | old | −0.81 | −2.06 (−0.078) | 0.06 | 2.05 | −7.23 | 3.66 | 3 |
| JM (−0.787) | minimal | 0.89 | 0.24 (−0.644) | 0.67 | 1.08 | 0.11 | 0.22 | 0 |
| JM (−0.787) | **final (P11)** | 0.86 | 0.80 (−0.644) | 0.88 | 0.33 | 0.77 | 0.03 | 0 |
| IES (−0.591) | old | 0.70 | 0.70 (−0.591) | 0.91 | 2.09 | 0.62 | 0.06 | 0 |
| IES (−0.591) | minimal | 0.77 | 0.77 (−0.591) | 0.93 | 1.84 | 0.75 | 0.04 | 0 |
| IES (−0.591) | **final (P11)** | 0.97 | 0.96 (−0.484) | 0.98 | 0.63 | 0.96 | 0.00 | 0 |
| EG (−0.739) | old | 0.34 | 0.34 (−0.739) | 0.86 | 2.97 | 0.00 | 0.32 | 0 |
| EG (−0.739) | minimal | 0.61 | 0.61 (−0.739) | 0.90 | 2.77 | 0.39 | 0.16 | 0 |
| EG (−0.739) | **final (P11)** | 0.96 | 0.96 (−0.739) | 0.97 | 1.53 | 0.95 | 0.01 | 0 |
| CH (−0.737) | old | 0.92 | 0.92 (−0.737) | 0.97 | 5.66 | 0.89 | 0.02 | 0 |
| CH (−0.737) | minimal | 0.99 | 0.96 (−0.326) | 0.98 | 5.68 | 0.94 | 0.01 | 0 |
| CH (−0.737) | **final (P12)** | 0.99 | 0.97 (−0.326) | 0.99 | 3.58 | 0.97 | 0.00 | 0 |
| HM (−0.789) | old | 0.67 | 0.67 (−0.789) | 0.91 | 0.91 | 0.55 | 0.09 | 0 |
| HM (−0.789) | minimal | 0.85 | 0.85 (−0.789) | 0.93 | 0.93 | 0.83 | 0.04 | 0 |
| HM (−0.789) | **final (P11)** | 0.92 | 0.90 (+0.195) | 0.96 | 0.61 | 0.89 | 0.01 | 0 |

**Totals over all 6 subjects** (144 single-run R² values each): old **6** negative, minimal **0**, final **0**.

P11 = final with auto-gain on. P12 = final with auto-gain switched off automatically, which only happens for CH.

(I left pooled R² out on purpose. JM's old target had that bogus ~12 Nm offset at −0.078, which makes the old torque look like it moves way more than it really does, so the old pooled R² looks better than it deserves. MSE doesn't have this problem.)

### What changed in the pipeline

| Step | Old pipeline | New pipeline |
|---|---|---|
| EMG envelope | demean → band-pass 30-300 Hz → rectify → low-pass 2 Hz → ÷ session-1 max | same, **then causal auto-gain per muscle** (off if pushes go both ways) |
| Passive map | auto-detected from passive trials with a 20 Nm sanity filter (let a bit of a movement trial in for JM) | **static holds only**: the still part at the start of each passive trial (YES: worksheet values in both) |
| Passive subtraction | target = torque − P(angle) | target = torque − **(α·P(angle) + β)**, α and β fitted on session-1 rest |
| Model input | 20 × 5 window | same 20 × 5 window |
| Model | LSTM 8-8, dropout 30%, 1 output (1,001 params) | **LSTM 16-16, dropout 20%, shared output + 1 linear expert per angle** (3,665 params) |
| Loss | plain MSE in Nm² (big angles dominate) | MSE on **per-angle scaled** targets (every angle counts the same) |
| Training settings | Nadam, lr 0.003, batch 8, at most 200 epochs, early stop 25 | same (screening used batch 64, max 150 epochs) |
| What I report | R² per angle | R² per angle **+ MSE / RMSE**, mean of 3 seeds |

### Bottom line

- **Chosen solution:** the 4-change final pipeline (`--recommended`). It gets rid of every negative R², improves every subject's pooled error, and makes results repeatable across seeds.
- **Plan B:** the 3-change minimal fix. Smaller, also no negative R², and better at YES −0.486 alone, but it doesn't solve JM's EMG drift.

---

## 9. Limitations

- **I chose the method by looking at session-2 results.** About 40 configurations were scored on session 2 and I picked from them. So the numbers for YES and JM are probably a bit optimistic (my guess is around 0.05-0.1 R² at the most-looked-at angles). The subjects that weren't used to pick the method (IES, EG and HM, with IES the cleanest) say the improvement is real, though.
- **Only 3 seeds.** Enough to catch the big seed problems, but the spread estimates are rough.
- **Only one retest session per subject.** I can't tell how this behaves in a third session.
- **YES −0.486 is still weak.** 0.20 with the final pipeline, 0.47 with the minimal fix. Within-trial passive relaxation (cause 2) isn't fixed, and with a 0.40 Nm signal there isn't much room.
- **Fast training protocol.** Most numbers use batch 64 and at most 150 epochs. The original batch-8 setup was only confirmed with 1 seed.
- **The EMG smoothing cheats a little, in both pipelines.** The 2 Hz smoothing filter runs forwards *and* backwards over the whole trial (so it adds no delay), and the offset removal uses the whole trial's average. So each sample has already "seen" a bit of the future (~0.1-0.5 s). That's true for old and new, so the comparison is fair, but none of these R² values could be hit live yet. The auto-gain itself only uses the past.
- **Auto-gain assumes one push direction.** It's switched off for subjects who push both ways (CH). For those subjects the EMG drift problem isn't handled.
- **JM's "before" numbers are noisy.** The old pipeline was very seed-unstable for JM (e.g. −0.787: 0.61 / 0.89 / −3.93), so its means swing a lot. Look at the single runs (Figure 16), not just the mean.

---

## 10. What I'd do next time

Mostly recording protocol stuff, since most of the problems came from the data, not the model:

- **Passive holds before AND after each active trial**, at every angle. Passive torque relaxes during a trial and gets "loosened" by the contractions, so one hold before the trial isn't enough.
- **A passive hold at every angle used in the pushing trials, in both sessions.** YES's session 1 has no clean static hold at −0.49 (so YES relies on hand-written worksheet values), and JM's −0.07 entry came from a movement trial.
- **Keep the push direction the same** at each angle in both sessions (CH flipped at −0.33 rad).
- **A short calibration contraction at the start of each session**, so EMG can be normalised per session. That's a cleaner alternative to the auto-gain.
- **Report RMSE (in Nm) next to R².** At the most plantarflexed angles the torque std is only 0.4-0.8 Nm, so R² on its own is misleading.
- **Use 3-5 seeds** and report the mean (or median), never a single lucky run.

---

## 11. Glossary and where things live

### Glossary

| Term | Plain meaning |
|---|---|
| Plantarflexion / dorsiflexion | Toes down / toes up. Here negative angle = plantarflexion. |
| Isometric | Pushing against something that doesn't move, so the joint angle stays fixed. |
| Session 1 / session 2 | The two recording sessions (both in the same recording file). Session 1 (lab name "test") = training data, session 2 (lab name "retest") = test data. |
| EMG | Electrical activity of a muscle, measured with skin electrodes. |
| Envelope | Smoothed "how active is the muscle" curve made from raw EMG. |
| MVC | Maximum voluntary contraction, i.e. pushing as hard as you can. |
| Nm | Newton-metre, the unit of torque. |
| rad / mrad | Angle units. 1 rad ≈ 57°, 1 mrad = 0.001 rad ≈ 0.06°. |
| std | Standard deviation: how much a signal typically wiggles around its average. |
| Variance | std². |
| Percentile | The 10th percentile is the value that 10% of the samples are below. |
| Passive torque | The torque the ankle makes when fully relaxed (stretched tissue pulling back). |
| Passive map | Lookup table: ankle angle → relaxed torque, from the passive trials. |
| Passive correction | Subtracting α·P + β instead of P; α and β fitted on session-1 rest periods. |
| Active torque | Measured torque minus passive torque, i.e. what the muscles make. This is the target. |
| R² | 1 = perfect, 0 = as good as guessing the average, negative = worse than that. |
| MSE / RMSE | Average squared error (Nm²) / its square root, the typical error size (Nm). |
| Pooled MSE | MSE over all 8 angles thrown together. |
| LSTM / GRU | Neural networks that read a sequence step by step with a small memory. |
| MLP | Plain neural network with no notion of time order. |
| TCN / 1D-CNN | Network that slides small filters along time. |
| Ridge regression | A plain linear (straight-line) model with a small penalty to keep the weights reasonable. |
| Dropout | Randomly switching off neurons during training so the network doesn't memorise. |
| Overfitting | Memorising the training data: great there, bad on new data. |
| Validation set | Session-1 data held back to decide when to stop training. |
| Test set | Data the model never sees in training (here: the whole of session 2). |
| Epoch | One full pass over the training data. |
| Learning rate | How big each weight-update step is. |
| Early stopping | Stop training when the validation error stops improving and keep the best version. |
| Batch size | How many samples the model looks at before each weight update. |
| Seed | Random number that sets the starting weights and the shuffling. Different seed = slightly different model. |
| Sandbox / real code | My quick experiment setup / the actual `train.py` pipeline. |
| Fast protocol | Batch 64 instead of 8 and at most 150 epochs, so testing doesn't take forever. |
| L2 penalty | An extra cost for big weights, so they stay small. |
| Zero-phase filter | A filter run forwards and backwards, so it adds no delay but uses a bit of the future. |
| Causal | Only uses past and present data, never the future, so it could run live. |
| Auto-gain (AGC) | Rescaling each EMG channel by its own recent range so amplitude changes cancel out. |
| Mixture of experts (MoE) | A shared network plus small specialised outputs, here one per ankle angle. |

### Where things live

- **This report:** `notes/report/dorsiflexion_report.md`, figures in `notes/report/figures/`.
- **Earlier, more technical notes:** `notes/dorsiflexion_investigation.md`.
- **Results:** `ML/experiments/results/` (sandbox runs) and `ML/experiments/results/pipeline/` (real-code runs). Summaries: `python -m ML.experiments.report` and `python -m ML.experiments.final_report`.
- **New code:** `ML/preprocessing/active_torque.py` (static map, passive correction), `ML/preprocessing/emg_agc.py` (auto-gain), `ML/experiments/` (sandbox, plugins, runners).
- **Changed code:** `ML/preprocessing/dataset_builder.py`, `ML/models/lstm.py`, `ML/training/__init__.py`, `ML/training/train.py`, `ML/config.py`.
- **How to run:**

  ```bash
  # final pipeline
  python ML/training/train.py --subject YES --recommended
  # old pipeline, exactly as before
  python ML/training/train.py --subject YES --legacy-passive --no-passive-gain
  ```

- **Status:** all of this is **uncommitted** in the worktree on branch `claude/dorsiflexion-r2-drop-341137`. I haven't decided yet what to do with the code.

---

## Appendix: all the screening runs

These are the sandbox runs (plus the ridge models). They're only comparable with each other, not with the real-code table in section 5. Mean R² of 3 seeds on session 2 (ridge has no randomness, so it's 1 run). "—" = not run for that subject.

**Fair?** = "yes" if no torque from the held-out session-2 active trials is used for the inputs, for training, or to adjust the target. (Every variant, the old one included, still uses session 2's own passive-hold map to define the session-2 target.) "**no**" = it uses session 2's measured torque, so it's cheating. Everything in tables A-D is fair; table E is not.

#### Starting point

| What it does | YES −0.486 | YES −0.381 | JM −0.644 | JM −0.360 | Code name |
|---|---:|---:|---:|---:|---|
| Original pipeline, untouched (sandbox version) | −1.09 | 0.78 | −0.04 | 0.41 | `baseline` |

#### A. Data / target fixes

| What it does | YES −0.486 | YES −0.381 | JM −0.644 | JM −0.360 | Code name |
|---|---:|---:|---:|---:|---|
| Clip EMG at 1.2× the training max (kills electrode spikes) | −1.30 | 0.79 | −0.04 | 0.49 | `clip12` |
| Feed the set-point angle instead of the measured angle | −0.76 | 0.80 | −0.03 | 0.61 | `knotpos` |
| Angle only, no EMG at all (leak test, just a diagnostic) | −0.87 | −0.07 | — | — | `posonly` |
| EMG auto-gain + angle weights, no passive fix | — | — | 0.79 | 0.91 | `gain_snr_agc_posw` |
| Passive correction + weights + drift-free training labels | 0.33 | 0.72 | 0.25 | 0.46 | `H_pgain_smooth_posw` |
| Same + set-point angle | 0.34 | 0.74 | 0.11 | 0.53 | `H_pgain_smooth_posw_knot` |

#### B. Loss / training fixes

| What it does | YES −0.486 | YES −0.381 | JM −0.644 | JM −0.360 | Code name |
|---|---:|---:|---:|---:|---|
| Angle-weighted loss only (every angle counts the same) | −1.16 | 0.77 | 0.12 | 0.31 | `posw_invvar` |
| Passive correction + angle weights (= minimal fix, sandbox version) | 0.38 | 0.76 | 0.18 | 0.75 | `H_pgain_posw` |
| Passive correction + drift-free labels + each angle's target shifted to average 0 and scaled to std 1 | 0.14 | 0.80 | 0.10 | 0.32 | `H_pgain_smooth_posnorm` |

#### C. Model changes

| What it does | YES −0.486 | YES −0.381 | JM −0.644 | JM −0.360 | Code name |
|---|---:|---:|---:|---:|---|
| Passive correction + zero-anchored LSTM (output forced to 0 when EMG is silent) | 0.35 | 0.71 | — | — | `H_anc_pgain_smooth_floortrain_posw` |
| Mixture-of-experts output only, no data fixes | −1.09 | 0.90 | 0.21 | 0.53 | `ml_moe` |
| Passive correction + mixture-of-experts output | 0.51 | 0.84 | 0.19 | 0.54 | `H_pgain_moe` |

#### D. Other model families on the minimal-fix data, plus a straight-line model

| What it does | YES −0.486 | YES −0.381 | JM −0.644 | JM −0.360 | Code name |
|---|---:|---:|---:|---:|---|
| My LSTM 8-8 (same data, for comparison) | 0.38 | 0.76 | 0.18 | 0.75 | `H_pgain_posw` |
| Plain feed-forward net on the flattened 200 ms window | 0.47 | 0.57 | 0.29 | 0.65 | `F_mlp` |
| Small causal 1D-CNN (TCN) | 0.26 | 0.74 | 0.03 | 0.69 | `F_tcn` |
| GRU 8-8 (same size as the LSTM, different cell) | 0.29 | 0.80 | 0.36 | 0.72 | `F_gru8` |
| Bigger LSTM 32-32 | 0.39 | 0.74 | 0.12 | 0.69 | `F_lstm32` |
| Linear ridge regression, original data | −0.55 | 0.55 | 0.37 | −0.46 | ridge A |
| Ridge, bug-fixed passive map | −0.55 | 0.55 | 0.37 | −0.46 | ridge B |
| Ridge, bug fix + passive correction | 0.33 | 0.37 | 0.35 | −0.47 | ridge C |
| Ridge, C + EMG auto-gain | 0.36 | 0.76 | 0.70 | 0.82 | ridge D |

(Ridge A and B are identical here on purpose. YES uses hand-entered worksheet passive values, so the map bug can't touch it, and JM's bug only hits −0.078, which isn't a column here. At JM −0.078, ridge goes 0.25 → 0.87 with the bug fix, and JM's pooled MSE goes 1.19 → 0.96.)

#### E. Cheating-ish variants I rejected (they use the session-2 torque)

| What it does | Fair? | YES −0.486 | YES −0.381 | JM −0.644 | JM −0.360 | Code name |
|---|---|---:|---:|---:|---:|---|
| Re-zero each session-2 trial on its own rest torque (constant) | **no** | −0.97 | 0.40 | 0.42 | 0.80 | `rezero_const` |
| Re-zero on rest torque, drifting baseline | **no** | −1.05 | 0.58 | 0.28 | 0.73 | `rezero_drift` |
| Re-zero drift + angle weights | **no** | 0.58 | 0.58 | 0.35 | 0.73 | `drift_posw` |
| Re-zero drift + zero-anchored LSTM | **no** | 0.66 | 0.70 | 0.36 | 0.55 | `anc_floor_drift` |
| Re-zero const + zero-anchored LSTM | **no** | 0.32 | 0.68 | 0.49 | 0.56 | `anc_floor_const` |

(A few tiny side-variants, like slightly different despike or MoE settings, are only in the raw result files and aren't shown here.)
