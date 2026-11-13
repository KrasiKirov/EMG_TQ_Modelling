# Final code / pipeline: how it works, how it's scored, and why

*A tutoring-style walk-through of the final version of the project, written so I can explain every step myself. Each step says what we did, why we did it, and what would happen without it.*

- **The recommended configuration is called `FINAL`.** It lives in `ML/research/experiments.py` and is run with `python -m ML.research.run ... --config FINAL`.
- In report 2 it's called **"E22 + auto rule"**: the report-1 final pipeline, plus rest-anchored labels, plus the "auto" switch for the auto-gain.
- How it was chosen, with all the numbers, is in [report 2](../report2/round2_report.md). This document is about **understanding** it.

> **How to read this.**
> - Start with **Part 0** (the 2-minute version).
> - Then go through Parts 1–11 in order. Every part ends with a short box (*what we did / why / remember*), and most have **check-yourself** questions with hidden answers (click to open).
> - Each part is tagged **[data processing]** (normal programming, nothing is learned), **[machine learning]** (the computer learns from examples) or **[scoring]** (checking how good the predictions are).
> - Part 15 lists questions a professor is likely to ask, with answers.

**Contents:**
[0. The 2-minute version](#part-0--the-2-minute-version) ·
[1. The problem](#part-1--the-problem-what-are-we-predicting) ·
[2. The data](#part-2--the-data-data-processing) ·
[3. EMG envelope](#part-3--from-raw-emg-to-an-envelope-data-processing) ·
[4. Trim + downsample](#part-4--trim-and-downsample-data-processing) ·
[5. Normalising EMG](#part-5--normalising-the-emg-data-processing) ·
[6. The target](#part-6--the-target-what-is-the-right-answer-data-processing) ·
[7. Windows](#part-7--windows-chopping-trials-into-examples-data-processing) ·
[8. Splitting](#part-8--splitting-train--validation--test-machine-learning-practice) ·
[9. The model](#part-9--the-model-machine-learning) ·
[10. Training](#part-10--training-machine-learning) ·
[11. Scoring](#part-11--scoring-how-good-is-it-scoring) ·
[12. Results](#part-12--the-results-and-what-they-mean) ·
[13. ML vs programming](#part-13--what-is-machine-learning-and-what-is-just-programming) ·
[14. How to run it](#part-14--how-to-run-it) ·
[15. Professor questions](#part-15--questions-a-professor-might-ask) ·
[16. Honest weak points](#part-16--honest-weak-points-of-the-code) ·
[Glossary](#glossary)

---

## Part 0 — The 2-minute version

**The idea.** We put sensors on 4 lower-leg muscles. They measure EMG, the tiny electrical signal a muscle makes when it works. We want a program that looks at those signals (plus the ankle angle) and says how much **force the person is voluntarily producing at the ankle**, measured as ankle **torque** (twisting force) in Nm.

**How:**
1. **Clean the EMG** into smooth "how active is this muscle" curves.
2. **Rescale them**, because the signal size changes from one recording session to the next.
3. **Work out the correct answer** (the voluntary torque) for every moment.
4. **Cut the recordings into 0.2-second pieces.**
5. **Train a small neural network** to go from a piece of EMG to the torque. It's an LSTM with one small "expert" per ankle angle.
6. **Test it on the other recording session** (the lab's "retest" block), which it never learned from.

**One model per person.** Each subject gets their own model, trained only on their own recordings (`run.py --subject YES` loads only YES). There's no single model for everybody. Pre-training on the other people first (config E20) made things worse.

**Result.** On the 3 main subjects (YES, JM, EG) the average score is R² ≈ 0.925 (1 = perfect), against 0.54 for the old version. At the hard "toes-down" angles it goes from ≈ 0 to ≈ 0.84.

![The final pipeline, step by step](figures/fig_final_pipeline.png)

*Figure 0: the whole pipeline. Grey = plain data processing, orange = machine learning, blue = scoring. The small grey text in each box is the code file that does that step.*

### The 4 switches of FINAL, and the other configurations named in this document

The code builds every experiment from 4 choices. In `experiments.py`, FINAL is `dict(passive='restanchor', emg='agc_auto', model='moe', balance='scale')`:

| switch | FINAL uses | in words | code | Part |
|---|---|---|---|---|
| `passive` (how the "right answer" is built) | `restanchor` | target = measured torque − relaxed level found inside the same trial | `run.restanchor_targets`, `truth.py` | 6 |
| `emg` (how EMG is rescaled) | `agc_auto` | ÷ training maximum, then auto-gain, unless the person pushes both ways | `emg_norm.normalise` | 5 |
| `model` | `moe` | 2 LSTM layers of 16 + one "expert" per angle | `models.build` → `lstm.build_model` | 9 |
| `balance` (how angles are weighted) | `scale` | divide each target by its angle's typical swing (σ) | `train.fit_predict` | 10 |

| name | passive | emg | model | balance | what it is |
|---|---|---|---|---|---|
| E01_old | legacy (old passive map) | ÷ max only | lstm8 | none | the original pipeline |
| E02_minimal | gain | ÷ max only | lstm8 | weights | report 1's minimal fix |
| E03_final | gain | auto-gain (always on) | moe | scale | report 1's final pipeline |
| E19 | gain | auto-gain | lstm8 | weights | small LSTM + auto-gain |
| E22 | restanchor | auto-gain (always on) | moe | scale | E03 + rest-anchored labels |
| E31 | restanchor | auto-gain | lstm8 | weights | small model + rest-anchored labels |
| E32 | restanchor | ÷ max only (auto-gain off) | moe | scale | E22 without auto-gain |
| **FINAL** | restanchor | **agc_auto** | moe | scale | = E22 for people who push one way, = E32 for people who push both ways |

*"gain" = the old passive map × α + β, fitted on the training session's relaxed moments (report 1). "lstm8" = the original small LSTM. "weights" = angle balancing with loss weights instead of scaled targets (same idea as "scale").*

---

## Part 1 — The problem: what are we predicting?

### The experiment, in normal words

- A person sits with their foot strapped to a plate. The plate is **locked** at one ankle angle, so the foot can't move. The person pushes against it (toes down, like pressing a gas pedal) and relaxes, again and again, for about 90 seconds. That's one **trial**.
- There are **8 trials per session**, one per ankle angle. Every person did **2 sessions**, one after the other in the same visit: S1 (the lab calls it "test") and S2 ("retest"). Both are stored in the same recording file.
- During each trial we record, 1000 times per second:
  - the **ankle angle**, in radians (rad; 1 rad ≈ 57°, so −0.486 rad ≈ 28° toes-down; negative = toes down = **plantarflexion**);
  - the **torque** the foot puts on the plate (Nm);
  - the raw **EMG** of 4 muscles: GM and GL (the two halves of the big calf muscle), SOL (the deeper calf muscle) and TA (the shin muscle that lifts the toes).

### What the program has to do

**Input:** the EMG of the 4 muscles + the ankle angle. **Output:** the torque the muscles are making *right now*.
- This kind of task is called **regression**: predicting a **number** (like "−12.3 Nm"), not a category (like "cat / dog").
- The inputs are called **features**: the information we give the model.
- The output we want is called the **target**: the right answer the model should learn to give.

**Analogy.** It's like guessing how hard someone is pressing a scale just by listening to how loud their muscles "hum". Louder hum → harder press. But how loud the hum is for the same press changes with the angle and from one session to the next. That's what makes it hard.

**Why do this at all, if we need the torque sensor to make the answers?** The sensor is needed once, for one training session per person. After that, the idea is to estimate the torque from EMG alone. "Learn on session 1, test on session 2" simulates exactly that.

### Why the toes-down (plantarflexed) angles were the problem

When the foot is pointed down, the calf muscle is already short and weak, so people can only produce a little torque there. For subject YES at −0.486 rad, the voluntary torque swings by only about **1 Nm**. At strong angles it swings by 30 Nm or more. A mistake of a few tenths of a Nm is huge compared with a 1 Nm signal, which is why the scores collapsed at those angles in the old version.

> **What we did:** framed it as a regression problem: 4 EMG signals + angle → voluntary ankle torque, one model per person.
> **Why:** after one session with the sensor, estimate torque from EMG alone.
> **Remember:** at toes-down angles the signal is tiny (~1 Nm), so small errors matter a lot.

<details><summary><b>Check yourself:</b> is this classification or regression? What are the features and what is the target?</summary>

Regression, because we predict a number (torque in Nm). The features are the 4 EMG signals + the ankle angle. The target is the voluntary (active) torque.
</details>

---

## Part 2 — The data [data processing]

**What the code does** (`ML/research/data.py`, function `load_subject`):

1. **Read the recording file.** It reads the subject's `.flb` file with the project's existing reader (`ML/preprocessing/flb_reader.py`). One file contains *all* trials of both sessions.
2. **Sort the trials into kinds:**
   - **active trials:** the pushing trials we learn from;
   - **passive trials:** the person stays relaxed while the plate holds or moves the foot. The *old* version used these to estimate the passive torque. FINAL still loads them but **doesn't use them** (Part 6);
   - **MVC trials:** "push as hard as you can". FINAL only uses these to put errors into perspective (as % of the maximum).
3. **Label each active trial** with its session (S1/S2) and angle.
4. **Save everything in a cache file**, so loading is fast the next time.

**What comes out:** a "bundle" (a Python dictionary). For every active trial it holds the processed EMG, the angle, the torque and the time axis (Parts 3–4 explain the processing).

**Why:** this is plain bookkeeping; nothing is learned. If trials were mislabelled (wrong session, wrong angle), everything after would be wrong. That's why the code reuses the original pipeline's trial-sorting rules, including some hand-written fixes for subjects whose trial comments are mislabelled (`TRIAL_OVERRIDES` in `ML/config.py`).

> **What we did:** loaded and sorted the recordings.
> **Why:** to know which trial is which (session, angle, type).
> **Remember:** 8 active trials per session × 2 sessions per person. One trial = one angle = ~90 s.

---

## Part 3 — From raw EMG to an "envelope" [data processing]

Raw EMG looks like noisy static: it wiggles up and down very fast around zero (Figure 1, panel 1). What matters is **how big** the wiggles are, not their exact shape. So we turn it into a smooth "how active is this muscle" curve, called the **envelope**.

**What the code does** (`ML/preprocessing/emg_envelope.py`, function `extract_envelope`), for each of the 4 muscles:

| step | what | why | without it |
|---|---|---|---|
| demean | subtract the average | a standard safety step (it copies the lab's MATLAB `demean.m`) | almost no difference here, because the next step removes a constant offset anyway |
| band-pass 30–300 Hz | keep only wiggles between 30 and 300 per second | muscle activity lives there. Slower wiggles are cable movement, faster ones are electrical noise | movement and noise would look like muscle activity |
| rectify | take the absolute value | the filtered signal is half negative, and its *size* is what we care about | the average of the raw signal is ≈ 0 whatever the effort |
| low-pass 2 Hz | smooth heavily | keeps only slow changes: the "effort level" | the curve would still be spiky |

**Tiny example.** Take 5 raw samples: +0.10, −0.12, +0.09, −0.11, +0.10 (mV).
- Their average is ≈ 0.01, which wrongly suggests almost no activity.
- After the absolute value they become 0.10, 0.12, 0.09, 0.11, 0.10, and the smoothed value is ≈ 0.10: the real activity level.

![Signal processing stages](figures/fig03a_signal_stages.png)

*Figure 1 (from report 1): real data from YES. Panels 1 → 2b are this part, 3a–3b are Part 5, and 4a–4b are Part 6. Panel 4b shows the **old** target, whose resting level isn't exactly 0. The final pipeline fixes that (Part 6).*

**Honest detail: the filters look into the future.** The filters are run forwards *and* backwards ("zero-phase"), so the curve isn't delayed. The price is that each envelope value is an average of the signal around it, and about half of that average comes from the *future*, up to roughly a quarter of a second ahead.
- You can see it in Figure 1, panel 2b: the envelope starts rising ~0.1 s before the push.
- That's fine for analysing recordings afterwards.
- A real-time device would need one-directional filters, which would delay the envelope by roughly 0.1–0.2 s. How much accuracy that would cost hasn't been tested (Part 16).

> **What we did:** filter → absolute value → smooth, giving one envelope per muscle.
> **Why:** the model needs "how active", not raw electrical noise.
> **Remember:** envelope = smoothed size of the EMG (30–300 Hz band-pass, 2 Hz low-pass). It is computed offline and uses a little future signal.

<details><summary><b>Check yourself:</b> why can't we just average the raw EMG?</summary>

Raw EMG goes up and down around zero, so its average is ≈ 0 even when the muscle works hard. We first take the absolute value (the size), then smooth.
</details>

---

## Part 4 — Trim and downsample [data processing]

**What the code does** (`data.py`, `_process_active`):
- **Drop the first 1 s of each trial.** Filters produce junk at the very start of a signal, so we cut it off.
- **Go from 1000 to 100 samples per second.** The envelope is smooth (2 Hz) and doesn't need 1000 points per second. The code first applies an "anti-alias" filter, then keeps one sample in ten; without the filter, keeping 1 in 10 can create fake wiggles. This filter is also zero-phase.
- The angle and torque are downsampled the same way, so everything stays aligned in time.

**Why:** 10× less data means much faster training, and nothing is lost, because nothing in a 2 Hz curve changes faster than that. **Result:** a ~90 s trial becomes ~8,900 time steps, each with 4 envelopes, 1 angle and 1 torque.

> **Remember:** from here on, 100 samples per second; one "step" = 10 ms.

---

## Part 5 — Normalising the EMG [data processing]

### The problem

The raw size of an envelope depends on things that have nothing to do with effort: electrode contact, skin, fat thickness, and so on. Two consequences:
- **The 4 muscles have very different sizes.** In Figure 1, GM is ≈ 6× bigger than TA.
- **The same muscle can have a different size in the other session.** For JM, the calf-muscle EMG in session 2 was only ≈ 0.36× (at −0.64 rad), 0.43× (at −0.36 rad) and 0.69× (at −0.79 rad) of its session-1 size for about the same torque. For YES at −0.38 rad it went the other way: ≈ 1.3× bigger (report 1, cause 4).

**Normalisation** means rescaling numbers so they're on a comparable scale.

> **Three statistics words used from here on.**
> - **Median:** sort the numbers and take the middle one. For 6.1, 6.2, 6.3, 6.4, 9.9 the median is 6.3 (the average is 6.98). One odd value barely moves the median.
> - **Percentile:** the p-th percentile is the value below which p% of the numbers fall. The 10th percentile of a muscle's envelope is roughly its "quiet" level; the 95th is roughly its "pushing" level. Unlike the min and max, a single spike can't move them much.
> - **Standard deviation (σ, "sigma"):** the typical distance of numbers from their average. 9, 11, 9, 11 has average 10 and σ = 1. A trial with σ = 15 Nm swings a lot; one with σ = 0.4 Nm barely moves.

### Step 5a: divide by the biggest training value

For each muscle, divide by its largest value in the **training session**. Only the training session, so we never peek at the test session. Each muscle then goes roughly from 0 to 1.
- **Example:** GM peaks at 0.030 mV and TA at 0.005 mV. After dividing, a GM value of 0.015 becomes 0.5 and a TA value of 0.0025 also becomes 0.5. Both read "half of its maximum".
- **Why:** it puts the 4 muscles on the same scale. That matters directly when auto-gain is **off** (subject CH, and the old configurations), and for the push-direction check of the auto rule below.
- **Honest detail:** when auto-gain is on (5 of 6 subjects), this step changes almost nothing. Auto-gain divides by each muscle's own range anyway, so a constant factor cancels out. A check found the auto-gain output changed by less than 0.001 on average without it.

### Step 5b: auto-gain (adaptive gain control)

Dividing by a training maximum doesn't fix the **session-to-session** change: if the electrodes give a signal 2× bigger in the test session, every number is 2× too big. So we add **auto-gain** (`emg_norm.py`, function `_agc`, the same algorithm as `ML/preprocessing/emg_agc.py`). For each muscle, as time goes on:
- **floor** = the 10th percentile of the last 20 s, roughly "the level when relaxed";
- **top** = the 95th percentile of the last 45 s, roughly "the level when pushing";
- **new value** = (value − floor) / (top − floor), kept between 0 and 2.

It only uses the **past** (it's "causal"), so it could run live. Some details:
- **Priors.** The floor and top are always a mix of the running percentile and a typical value from the training data (the "prior"). At the start of a trial there's no past yet, so the prior dominates. Even later the prior keeps 20% of the floor and 10% of the top, so a gain change is removed *almost* but not completely.
- **Minimum range.** The range top − floor is never allowed below 0.35× the typical training range. Otherwise a nearly silent muscle would be blown up into noise.
- **Artefacts.** Sudden electrode "pops" are removed first.

**Analogy.** A microphone that automatically adjusts its volume to how loud the speaker has been in the last half minute. Whether they stand close or far, "talking normally" ends up at about the same level.

**Tiny example.**
- **Session 1:** relaxed = 0.10, pushing = 0.60. A value of 0.35 becomes (0.35 − 0.10)/(0.60 − 0.10) = **0.5**.
- **Session 2, electrodes give everything 2× bigger:** relaxed 0.20, pushing 1.20, and the same effort reads 0.70. Using only the running percentiles: (0.70 − 0.20)/(1.20 − 0.20) = **0.5**, the same answer.
- **With the 20%/10% prior mix the code really uses:** the floor is 0.18 and the top 1.14, giving ≈ 0.54. Most of the 2× change is gone, but not all of it.

**The catch (a likely exam question): auto-gain can't tell "the electrodes give 2× more signal" from "the person pushes 2× harder".** Both make the whole EMG bigger, and auto-gain divides both away.
- What survives: the ups and downs *within* the last ~45 s. A push half as strong as the recent strong pushes still reads about half.
- What's lost: the overall effort level of the whole trial. The torque size then comes from what the model learned at that angle in the training session.
- Why it still works here: both sessions had the same task (same angles, same kind of pushes). If one session had very different effort levels, this would be a problem.

**What if we didn't do this?** The original small model scores 0.826 without auto-gain and 0.917 with it (E02 vs E19). That's the single biggest improvement.

### The "auto" rule (it only matters for one subject)

Auto-gain throws away the *absolute* size of the signal. Usually that's good, but one subject (CH) pushed **toes-up** at some angles and **toes-down** at others. To tell which muscle group is really working, the model then needs the absolute sizes. So, before training:
1. The code measures the push direction of every training trial: torque during high activity minus torque during low activity (`push_directions` in `emg_norm.py`).
2. If some trials push more than +2 Nm one way and others more than 2 Nm the other way, **auto-gain is switched off** for that subject (option `agc_auto`). This is report 1's rule, written before this round.
3. Among the 6 subjects, only CH triggers it. Its training trials push about +9 to +28 Nm and −32 to −55 Nm, while all of YES's are negative.

New result files record the decision in `details.emg.auto_decision` ("AGC ON" or "AGC OFF (pushes in both directions)"), and the push directions in `details.push_directions`.

> **What we did:** ÷ training maximum, then auto-gain (unless the person pushes both ways).
> **Why:** make the muscles comparable, and remove most of the session-to-session change in signal size.
> **Remember:** auto-gain = "volume knob that adapts using only the past". It's the biggest single fix (+0.09 R²). It can't tell gain from effort.

<details><summary><b>Check yourself:</b> why compute the maximum and the priors from the training session only?</summary>

Because the test session must stay "unseen". If we used the test session's own statistics, information from the test data would leak into the inputs. That's called **data leakage**, and it makes results look better than they would be in real use. (Auto-gain does adapt to the test EMG while predicting, but only to *past* samples and never to the torque, exactly as a live device would.)
</details>

<details><summary><b>Check yourself:</b> why was MVC normalisation (EMG as % of maximum) not used?</summary>

It was tested (configs E10 and E11) and was worse than auto-gain: 0.887 and 0.904 vs 0.925 for E03. The MVC was recorded only once, at the start of the visit, so a fixed MVC scale can't follow the change in EMG size between the two sessions. Auto-gain re-adjusts as it goes.
</details>

---

## Part 6 — The target: what is the "right answer"? [data processing]

This is the part that changed most, and the one people get confused about.

### Measured torque = muscle part + passive part

The torque sensor measures **everything**. Even with the muscles completely relaxed, a stretched ankle pulls back a bit, like a rubber band. That's the **passive torque**. We only want the **active** (voluntary) part:

> **active torque = measured torque − passive torque**

**Example.** At YES −0.486 rad in session 2, the relaxed torque is about **+8.5 Nm** (Figure 2). When YES pushes, the sensor reads about +7.5 Nm, so the active torque is 7.5 − 8.5 = **−1.0 Nm** (negative = pushing toes down).

### How the old version estimated the passive part, and why that failed

The old version measured passive torque in **separate relaxed recordings** (the passive trials) and subtracted that value. But the relaxed torque *during* a pushing trial isn't the same:
- the tissue relaxes over time (YES −0.486, session 2: the hold just before the trial said 10.6 Nm, while relaxed moments inside the trial sat around 8.5);
- it drifts slowly during the 90 s;
- it differs between sessions.

So the "correct answer" itself was off by 1–2 Nm, at angles where the whole signal is only ~1 Nm. And one subject (IES) has no passive recordings at all.

### The fix: rest-anchored labels (`truth.py`, used by `restanchor_targets` in `run.py`)

**The idea:** when the person is relaxed, the measured torque *is* the passive torque. So we measure the passive torque inside the trial itself.

1. **Find the relaxed moments from the EMG only** (`truth.session_rest` → `active_torque.detect_rest`):
   - For each moment, compute the muscles' average activity above their quiet level.
   - If it drops below a small threshold (0.03), we count as "relaxed"; we count as active again once it goes above 0.06. Two thresholds stop it flickering on and off.
   - A relaxed stretch must last at least 0.5 s, and its first 0.3 s are skipped, because the torque needs a moment to settle.
   - The torque is **never** used to decide what counts as relaxed.
2. **Draw a smooth line through the torque at those relaxed moments** (`truth.baseline` → `active_torque.rest_baseline`):
   - Take the median torque of every relaxed stretch.
   - Each point of the line is a weighted average of those medians: stretches a few seconds away count most, and stretches more than ~16 s away hardly count (a Gaussian "bell" with σ = 8 s).
   - Stretches that look like outliers are ignored. Before the first and after the last stretch, the line stays flat.
3. **target = measured torque − that line.**

![Reference truth on YES](figures/truth_inspection_YES.png)

*Figure 2: YES, all 16 trials. Grey = measured torque, green dots = relaxed moments found from the EMG, black = the line through them (the passive torque inside the trial). Dashed blue = the static passive map (for YES, essentially what the old pipeline subtracted). Dotted yellow = the mean of the hold recorded just before each trial; the old pipeline never used it, it was tried separately as config E04. At the toes-down angles (top rows) both sit clearly above the real relaxed level.*

**Important:** only the **training** session's torque is used to make training labels. The test session's torque is never given to the model. (In Part 11 the same construction is applied to the test session, but only to **grade** the predictions.)

**What if we didn't do this?**
- On YES/JM/EG the score barely changes: E03 0.925 vs E22 0.925, because report 1's passive fix was already decent there. But the model predicts more fake torque when you're relaxed (0.25 vs 0.17 Nm, Part 11).
- On IES, which has no passive recordings, the toes-down angles collapse (R² ≈ 0.05) instead of reaching 0.98.

> **What we did:** active torque = measured − the passive level found during the relaxed moments of the same trial.
> **Why:** the passive torque inside a trial differs from separate passive recordings, and some subjects have none.
> **Remember:** relaxed moments are found from the EMG only; training labels use the training session only.

<details><summary><b>Check yourself:</b> why don't we use the torque to find the relaxed moments?</summary>

Because then the definition of "relaxed" would depend on the thing we're trying to measure. A slow push could be mistaken for "relaxed", and its torque would be wrongly subtracted. The EMG tells us independently whether the muscles are working.
</details>

<details><summary><b>Check yourself:</b> during relaxed moments, what should the target be?</summary>

About 0 Nm, because nobody is pushing. The rest-anchored labels make that true by construction.
</details>

---

## Part 7 — Windows: chopping trials into examples [data processing]

The model doesn't look at a whole 90 s trial at once. It looks at short pieces called **windows** (`splits.py`, function `windows`):
- one window = **20 steps = 200 ms**;
- at each step, **5 numbers**: GM, GL, SOL, TA (normalised) and the ankle angle;
- so one window is a little table of **20 × 5** numbers;
- its **answer** is the target torque at the **last** step of the window.

The window slides along one step (10 ms) at a time, so neighbouring windows overlap almost completely. A ~90 s trial gives ~8,900 windows.

**Why 200 ms of history and not just the current moment?** Force builds up a little *after* the muscle's electrical signal (tens of milliseconds). Seeing the recent past also tells the model whether the activity is rising or falling.

**Tensors.** All windows stacked together form a 3-D block of numbers with shape (number of windows, 20, 5). For YES's training data that's (56,896, 20, 5). A **tensor** is just the name for an N-dimensional array; a NumPy array of shape (56896, 20, 5) is exactly that.

![One model input](figures/fig03b_model_input.png)

*Figure 3 (from report 1): downsampling and cutting one 20 × 5 window.*

**A real example** (YES, training data): one window's first row is [1.196, 1.006, 1.143, 1.215, −0.279]: 4 auto-gain EMG values, then the angle (−0.279 rad). Its answer is −10.82 Nm.

> **What we did:** cut every trial into overlapping 200 ms windows of 20 × 5 numbers.
> **Why:** the model learns from many small examples and needs a bit of history.
> **Remember:** input shape (20, 5); answer = the torque at the last step.

---

## Part 8 — Splitting: train / validation / test [machine learning practice]

### Why split at all?

If you test a student with the exact questions they practised, they can score 100% by memorising without understanding. To check real understanding, you need **new** questions.
- **Training data:** the practice questions. The model learns from these.
- **Validation data:** a mock exam *during* training. The model doesn't learn from it; we watch its score to decide when to stop training (Part 10).
- **Test data:** the real exam: the other session. It's never used for training, for deciding when to stop, for the fixed EMG statistics, or for the training labels.

**Overfitting** means memorising the training examples instead of learning the real pattern: great on training data, bad on new data. **Underfitting** means not even learning the training data well, because the model is too simple or trained too little.

### Our split (`splits.py`, function `build`, protocol `S1toS2`)

- **Training session (S1):** in each trial, the first 80% of the windows → **train**, the last 20% → **validation**.
- **The other session (S2):** **every** window → **test**.
- Then everything is repeated the other way round (`S2toS1`: learn on S2, test on S1).
- For YES (`S1toS2`) that's 56,896 training, 14,232 validation and 71,128 test windows.

![Data split](figures/fig04_data_split.png)

*Figure 4 (from report 1): the split for `S1toS2`. The figure says 14,224 validation windows, from report 1's code. The current code gives 14,232: every window after the 80% cut goes to validation, 8 more in total. Train and test are unchanged.*

**Why test on the other session, and not on a random 20% of the same session?** In real use you'd train once and use the model later, when conditions have changed: electrode contact, fatigue, maybe electrode placement.
- Our second session is a later block of the same visit, so it's the closest stand-in we have. A true different-day test would probably be harder.
- Testing within the same session is much easier, and would give an over-optimistic score.

**Why the last 20% for validation and not random windows?** Neighbouring windows overlap 95%. If validation windows were scattered randomly, each would have a near-copy in the training data, and the mock exam would be too easy.

**Caveat for this project (be ready for it).** We compared 23 configurations on the test sessions of YES/JM/EG and picked the best from those scores. That's a form of choosing on the test data, so the winner's score there is a bit optimistic. Report 2 handles this with three extra checks:
- **cross-fitting:** pick on one direction, check on the other;
- **leave-one-subject-out:** pick on two people, check the third;
- **3 held-out subjects** (Part 11).

> **What we did:** learn on one session (80% train / 20% validation), test on the other session, in both directions.
> **Why:** to measure how well it works on a new session, not how well it memorises.
> **Remember:** train = practice, validation = mock exam for stopping, test = exam on the other session.

<details><summary><b>Check yourself:</b> what would go wrong if we used the test session to decide when to stop training?</summary>

We'd be tuning the model on the exam itself. The reported score would be too optimistic, because the test would no longer be "unseen".
</details>

---

## Part 9 — The model [machine learning]

### What is a neural network, in one paragraph?

A neural network is a big formula with many adjustable numbers inside it, called **parameters** or **weights**. You put numbers in (our 20 × 5 window) and a number comes out (the torque).
- At first the parameters are random, so the output is nonsense.
- **Training** (Part 10) adjusts them little by little, so the outputs get closer to the right answers.
- Our final model has **3,665** parameters, which is small. Big language models have billions.

### Why an LSTM?

Our input is a **sequence**: 20 steps in time order. An **LSTM** (Long Short-Term Memory) is a type of neural network made to read sequences one step at a time while keeping a small **memory**. At each step it updates that memory with the new values, deciding what to keep and what to forget. After the last step, the memory is a summary of the window (e.g. "calf activity has been rising for 100 ms").

**Analogy:** reading a sentence word by word and keeping the meaning in your head, instead of looking at all the letters at once.

- **"16 units"** means the memory is 16 numbers wide.
- Inside, an LSTM uses **activation functions**: small fixed formulas like tanh (squashes any number into −1…+1) and sigmoid (0…1). They let the network model curved, non-straight-line relationships; without them, stacking layers would still only give a straight-line formula.

### The final model, piece by piece (`ML/models/lstm.py`, `build_model(..., moe_knots=..., moe_expert_bias=False)`)

![Final model architecture](figures/fig06_arch_new.png)

*Figure 5 (from report 1): the model used by `FINAL`, E03 and E22: 3,665 parameters for 8 angles. The "was 8 / 30%" notes compare it with the old model.*

1. **LSTM, 16 units:** reads the 20 × 5 window and outputs 16 numbers at every step (1,408 parameters).
2. **Dropout 20%:** during training only, 20% of those numbers are randomly set to 0 at each update. This forces the network not to rely on any single number, which reduces overfitting. At prediction time nothing is dropped.
3. **Second LSTM, 16 units:** reads that and outputs one summary of 16 numbers (2,112 parameters). Then dropout 20% again.
4. **Shared output:** one weighted sum of the 16 numbers plus a constant → one value (17 parameters).
5. **Experts:** one extra weighted sum *per training angle*, with 16 weights each and **no constant**: 8 × 16 = 128 parameters. This is called a **mixture of experts**.
6. **Angle gate:** reads the ankle angle at the last step and picks that angle's expert. Between two training angles it blends the two neighbours.
7. **Output** = shared value + the chosen expert's value. This is in "scaled" units, and is multiplied back to Nm (Part 10).

**Why one expert per angle?** The relationship between EMG and torque changes with the angle: a short calf muscle makes less torque for the same activity. A single shared output would be fitted mostly to the strong angles; each expert adds a small correction for its own angle.
- The experts start at zero, so at the start the model behaves like a plain LSTM.
- A small penalty (L2, 0.001) keeps the expert weights small unless the data really needs them.

**Why no constant in the experts?** A constant per angle would let each expert learn "at this angle the torque is usually about X Nm". Report 1 dropped it because of CH: at one angle CH pushed toes-up in session 1 and toes-down in session 2, so a "usual torque" learned on one session is wrong on the other. The experts can only change how the 16 summary numbers are turned into torque.

**What if we removed the experts?** Honest answer: we never removed *only* the experts. The closest run is E31, which changes several things at once compared with FINAL/E22:
- the original tiny LSTM (8 units instead of 16, dropout 30% instead of 20%);
- no experts;
- angle balancing with loss weights instead of scaled targets (same idea).

Together these cost about 0.009 R² on YES/JM/EG (0.917 vs 0.925), and more on held-out HM (0.895 vs 0.930). E31 also sometimes overshoots big toes-down pushes (report 2, Figure 7d). So the bigger model with experts helps a little and is steadier, but how much of that is the experts alone wasn't tested.

**The old model, for comparison:** LSTM(8) → dropout 30% → LSTM(8) → dropout 30% → one output, 1,001 parameters ([figure](figures/fig05_arch_old.png)).

> **What we did:** two LSTM layers read the 200 ms window; a shared output plus an angle-specific expert give the torque.
> **Why:** the LSTM handles the time order; the experts handle "the EMG → torque relationship differs per angle".
> **Remember:** 3,665 parameters; LSTM = sequence reader with memory; dropout = random switching-off during training, against memorising.

<details><summary><b>Check yourself:</b> why does dropout only happen during training?</summary>

Dropout is a training trick to stop the network from relying on single numbers (memorising). When predicting we want the full, best answer, so nothing is dropped.
</details>

---

## Part 10 — Training [machine learning]

Training means repeating this loop many times:
1. show the model a group of windows;
2. it guesses the torques;
3. measure how wrong the guesses are;
4. adjust the parameters a little, to be less wrong.

The code is `train.py`, function `fit_predict`, using Keras `model.fit`.

### The loss: how wrong is the model?

The **loss** is one number that says how bad the guesses are. Here it's the **mean squared error (MSE)**: the average of (guess − right answer)².

**Example.** Three windows with right answers 10, 0 and −5, and guesses 8, 1 and −5.
- The errors are −2, +1 and 0; squared, 4, 1 and 0.
- MSE = (4 + 1 + 0) / 3 = **1.67**.

Squaring makes big errors count much more than small ones, and treats over- and under-shooting the same.

### Making every angle count the same ("scale")

With plain MSE there's a problem in our project. At YES's strong angles the torque swings with σ ≈ 15 Nm; at −0.486, only σ ≈ 0.64 Nm. Errors at strong angles are naturally much bigger, so plain MSE is dominated by them and basically ignores the toes-down angle. In squared units, the strong angle weighs about (15 / 0.64)² ≈ 550× more.

**What the code does:**
1. For each training angle, compute the torque's σ on the training data. For YES, from toes-down to toes-up: 0.64, 2.67, 4.57, 7.54, 8.51, 13.77, 15.05 and 12.96 Nm.
2. **Divide every target by its angle's σ** before training, so every angle's target has the same typical size (≈ 1).
3. After predicting, **multiply back** by σ to get Nm. Test windows use the σ of the closest training angle.

This was report 1's fix for "training basically ignored the weak angles".

### The optimizer: how the model adjusts itself

- **Gradient:** for each parameter, the computer works out "if I nudge this number up a little, does the loss go up or down, and by how much?" That set of directions is the gradient. Keras computes it automatically ("backpropagation").
- **Optimizer:** the rule that uses the gradient to adjust the parameters. We use **Nadam** (Adam with Nesterov momentum), a popular variant of gradient descent. It does two extra things:
  - it keeps some "momentum" from previous steps, like a ball rolling downhill;
  - it gives every parameter its own step size, based on how big that parameter's recent gradients were.
- **Learning rate (0.003):** how big each adjustment is. Too big and the model jumps around and never settles; too small and it learns painfully slowly. *Analogy:* how big a step a student takes when correcting a mistake.

### Batches and epochs

- **Batch size (64):** the model looks at 64 windows, averages their errors, then makes one adjustment. Bigger batches are faster and smoother; smaller ones are noisier.
- **Epoch:** one full pass over all training windows. For YES that's 56,896 / 64 ≈ 890 adjustments per epoch.
- **Why these "fast" settings:** the original project trained more slowly: batch 8, learning rate × 0.7 after 10 epochs without improvement, stop after 25, up to 200 epochs. We used the faster settings (below) to run hundreds of experiments in reasonable time.
  - Report 1 re-ran the old pipeline and report 1's final pipeline once with the original settings (seed 0, YES and JM) and got the same old-vs-new picture.
  - **FINAL itself was not re-run with the original settings** (Part 16).

### When to stop: validation, early stopping, learning-rate drops

After every epoch the code computes the loss on the **validation** windows (the mock exam):
- **Learning-rate drop** (`ReduceLROnPlateau`): if the validation loss hasn't improved for 6 epochs, halve the learning rate (smaller, more careful steps).
- **Early stopping** (`EarlyStopping`): if it hasn't improved for 15 epochs, stop, and go back to the parameters from the **best** epoch. The maximum is 150 epochs.
- **Real example:** FINAL/E22 on YES (seed 0) trained for 25 epochs. The best validation loss was at epoch 10, so epoch 10's parameters were kept. It took ~6 minutes on one CPU core.
- **Why:** this is the main protection against overfitting. When the mock-exam score stops improving while the training score keeps improving, the model has started memorising.

### Seeds: why we train 3 times

The starting parameters and the order of the batches are random. The **seed** is the number that fixes that randomness, so a run can be repeated exactly.
- **Main subjects (YES, JM, EG):** each setup is trained with seeds 0, 1 and 2, and we report the **average**. The 3 held-out subjects (IES, CH, HM) were run with seed 0 only.
- **We never pick the best seed.** That would be cheating, like retaking an exam 3 times and reporting only the best grade.
- **Overall, FINAL is stable:** its main score (the mean over 8 angles) varies by ~0.002–0.02 R² between seeds (standard deviation; the largest gap between two seeds is 0.04).
- **But not at the hardest angle:** at YES −0.486 (S1 → S2) the 3 seeds give R² 0.47, 0.41 and 0.14 (mean 0.34). When the whole signal is ~1 Nm, small random differences in training matter a lot, which is another reason to always report the mean.

**Hyperparameters.** All these settings (learning rate, batch size, number of units, dropout, patience, …) are **hyperparameters**: chosen by us, not learned by the model. Ours are in `ML/research/config.py` (`PROTOCOLS['fast']`) and `ML/models/lstm.py`.

> **What we did:** minimise the MSE on per-angle-scaled targets with Nadam, in batches of 64, stopping on the validation loss; 3 seeds.
> **Why:** learn the EMG → torque relationship without the strong angles dominating, and without memorising.
> **Remember:**
> - loss = how wrong; gradient = which direction to adjust; learning rate = step size;
> - epoch = one pass; early stopping = stop when the mock exam stops improving.

<details><summary><b>Check yourself:</b> what happens if you remove the per-angle scaling?</summary>

The loss is dominated by the strong angles, whose errors are ~550× bigger in squared units, so the model barely cares about the toes-down angles and gets them wrong.
</details>

<details><summary><b>Check yourself:</b> why keep the parameters from the best epoch instead of the last one?</summary>

After the best epoch the validation loss got worse: the model started memorising the training data. The best epoch is the version that worked best on data it didn't learn from.
</details>

---

## Part 11 — Scoring: how good is it? [scoring]

### Score per angle, on the other session

For each test trial (one subject, one angle, one session), the code compares the predictions with the right answers (`evaluate.metrics`, `truth.rescore`):
- **RMSE** (root-mean-square error) = √MSE. It's in Nm, so read it as "the typical error size".
- **R²** ("R-squared") = 1 − (sum of squared errors) / (sum of squared differences between the truth and its own average). In words: how much better the model is than always guessing the trial's average torque.
  - 1 = perfect;
  - 0 = no better than guessing the average;
  - **negative = worse than guessing the average**.

**Why R² is harsh at toes-down angles, in numbers:**

| case | typical swing of the truth (σ) | typical error (RMSE) | R² ≈ 1 − (error / σ)² |
|---|---|---|---|
| strong angle | 15 Nm | 2.5 Nm | **0.97** |
| toes-down angle | 0.4 Nm | 0.3 Nm (smaller!) | **0.44** |
| old version, toes-down | 0.4 Nm | 1.5 Nm | **−13** |

That's why we always report RMSE in Nm next to R².

### The reference truth (the "answer key")

To grade fairly, every model is compared with **the same** answer key: the rest-anchored active torque of Part 6, built on the **test** session.
- **Why:** the old way graded each model against its own estimate of the passive torque, so a model with a wrong passive estimate was graded against a wrong answer key. That hid a ~12 Nm error at JM −0.078, for example.
- The model never sees this answer key; it's only used for grading.
- **Honest caveat ("circularity"):** FINAL builds its *training* labels the same way as the answer key, so it's trained toward the same definition the grader uses. That's a built-in advantage, mainly on the "phantom torque" score below.

![Own target vs reference truth](figures/fig02_own_target_vs_truth.png)

*Figure 6 (from report 2): the old model graded against its own answer key (blue) vs the shared truth (black).*

### Combining the scores

1. **Floor at −1.** Each angle's R² (averaged over the seeds) is cut at −1, so one disastrous angle (e.g. −60) can't swamp everything.
2. **Mean over the 8 angles** gives one number per subject per direction: the "main score".
3. **Mean over** 3 subjects × 2 directions (S1→S2 and S2→S1).

**"Weak-2"** is the same calculation, but only for each subject's 2 most toes-down angles. That's the number that answers "did we fix the problem?".

### Phantom torque

The average size of the prediction during the relaxed moments of the test trial. It should be 0, because nobody is pushing. A model that "hallucinates" force when you're relaxed would be bad for any real use.

### How the winner was chosen (short version; details in report 2)

1. **The written selection rule** (`ML/research/results/SELECTION_RULE.md`): highest main score on YES/JM/EG with 3 seeds. It was written before any held-out result was opened, but after most screening results were in.
   - The rule picked **E29** (E03 with a 400 ms window instead of 200 ms): 0.932 vs 0.925 for E03 and E22. That gap isn't significant once you correct for making ~23 comparisons.
   - E29 only got its 2 extra seeds after its seed-0 run had beaten E03, so 0.932 is a bit optimistic.
2. **Cross-fitting** (pick on one direction, check on the other): E29 won both, so the cross-fitted score is also 0.932. Leave-one-subject-out gives 0.925.
3. **Then a check on 3 held-out subjects** (IES, CH, HM). E29 and E03 collapse on IES, the subject with no passive recordings, because their labels need a passive recording. So the recommendation was changed to E22 + the auto rule = `FINAL`.

**Say this openly:** FINAL is *not* what the pre-written rule picked; it was chosen after seeing the held-out subjects. So those subjects are no longer independent evidence for FINAL.

> **What we did:** per-angle R² and RMSE against one shared answer key, on the other session; averaged over angles, seeds and both directions.
> **Why:** a fair comparison, a realistic test, no lucky seeds.
> **Remember:** R² asks "better than guessing the average?" (1 = perfect, < 0 = worse). RMSE = typical error in Nm. Toes-down angles have small swings, so look at RMSE too.

<details><summary><b>Check yourself:</b> a model has R² = 0 at an angle. What does that mean?</summary>

It's exactly as good as always predicting that trial's average torque: it explains none of the ups and downs.
</details>

---

## Part 12 — The results and what they mean

**Main subjects** (YES, JM, EG; both directions; mean of 3 seeds; against the shared truth):

| | mean R² | R² at the 2 most toes-down angles | typical error there (RMSE) | phantom torque at rest |
|---|---|---|---|---|
| old pipeline | 0.540 | 0.04 | 1.27 Nm | 1.25 Nm |
| report-1 final (E03) | 0.925 | 0.85 | 0.44 Nm | 0.25 Nm |
| **FINAL** (= E22 here: auto-gain stays on for all 3) | **0.925** | **0.84** | **0.45 Nm** | **0.17 Nm** |

**Held-out subjects** (IES, CH, HM; seed 0 only):

| | mean R² | R² at the 2 most toes-down angles | phantom torque | R² without CH's flipped angle (same trials for all) |
|---|---|---|---|---|
| old pipeline | 0.874 | 0.70 | 1.07 Nm | 0.878 |
| report-1 final (E03, auto-gain always on) | 0.787 | 0.66 | 0.85 Nm | 0.870 |
| E22 (= FINAL without the auto switch) | 0.884 | 0.97 | 0.18 Nm | **0.967** |
| **FINAL** (E22; auto-gain off for CH) | **0.966** | **0.97** | **0.17 Nm** | — |

**How to read the held-out table:**
- **Why is E03 worse than the old pipeline here?** Two subjects explain it:
  - IES has no passive recordings, so E03's labels keep several Nm of passive torque (IES: E03 0.70 vs old 0.82).
  - CH pushed toes-up at one angle in session 1 and toes-down at the same angle in session 2. Every auto-gain model breaks there (R² floored at −1), which pulls CH from ~0.98 down to ~0.73.
- **The FINAL row isn't a fair head-to-head.** It's the only row that got the "auto-gain off for CH" switch; the old pipeline and E03 weren't re-run with it.
  - The fair comparison is the last column (same trials for everyone, CH's flipped angle removed): 0.967 vs 0.878 vs 0.870. So the fair gap to E03 is about 0.10, and it comes mostly from IES.
- **The FINAL numbers are real FINAL output.** `FINAL` was run on all 6 subjects in both directions (seed 0), and in all 12 runs it gave exactly the same predictions as the E22 (or, for CH, E32) runs used here. But FINAL was recommended after seeing these subjects, so they're not independent evidence for it.

**Your hardest angle, YES −0.486** (trained on S1, tested on S2):
- The old R² was −14.3 (error 1.52 Nm).
- FINAL gets **0.34** on average over the 3 seeds (error 0.31 Nm), and the predictions now follow the pushes (figure below).
- **Be ready for this question: at this one cell FINAL is the lowest of the new models.** E03 gets 0.47 (error 0.28 Nm), E29 0.42, E31 0.55. The difference is small in Nm (0.31 vs 0.28) and within FINAL's seed spread here (0.14–0.47). Over the two most toes-down angles of all 3 subjects, FINAL and E03 tie (0.84 vs 0.85).
- **Why R² is still lowish:** the person only pushes ~1 Nm there. The error is small next to their maximum force at that angle (~2% of it), but still about a third of the push itself.

![FINAL on YES, all 8 angles](figures/full_trial_FINAL_YES_S1toS2.png)

*Figure 7: FINAL (identical to E22 here), YES, trained on session 1, tested on session 2, all 8 angles, against the reference truth. This is **seed 0 only**, so its top panel (−0.486 rad) reads R² 0.47 / RMSE 0.28 Nm; the 3-seed average there is 0.34 / 0.31 Nm.*

---

## Part 13 — What is machine learning and what is just programming?

| step | file | type | learns anything? |
|---|---|---|---|
| read files, sort trials | `ML/research/data.py`, `ML/preprocessing/flb_reader.py` | programming / bookkeeping | no |
| EMG envelope, downsample | `ML/preprocessing/emg_envelope.py`, `data.py` | signal processing (fixed formulas) | no |
| normalise EMG, auto-gain, auto rule | `ML/research/emg_norm.py` | data processing (fixed formulas using statistics of the training data) | no |
| rest detection + target | `ML/research/truth.py`, `ML/preprocessing/active_torque.py`, `run.py` | data processing (fixed rules) | no |
| windows, splitting | `ML/research/splits.py` | data processing | no |
| model definition | `ML/research/models.py`, `ML/models/lstm.py` | **machine learning**: defines the formula with 3,665 adjustable numbers | it's the thing that learns |
| training | `ML/research/train.py` (Keras `model.fit`) | **machine learning**: adjusts the numbers | **yes, the only step that learns** |
| prediction | `model.predict` in `train.py` | machine learning (applies the learned numbers) | no |
| scoring, tables, figures | `evaluate.py`, `truth.py`, `analyze.py`, `tables.py`, `figures.py` | evaluation / programming | no |
| running many jobs | `run.py`, `queue.py` | programming (job management) | no |

**Key sentence for the presentation:** *only the training step learns. Everything before it prepares good inputs and good answers, and everything after it checks the result. Most of our improvement came from the preparation steps (auto-gain, correct answers), not from the model.*

---

## Part 14 — How to run it

All commands are run from the worktree folder, in a bash terminal:

```bash
source ML/experiments/env.sh        # sets $PY (the Python with TensorFlow) and where the data is
# one model: subject YES, learn on session 1, test on session 2, seed 0, with trial plots
$PY -m ML.research.run --study demo --subject YES --protocol S1toS2 --config FINAL --seed 0 --plot
# many models in parallel (9 at a time is safe on this PC)
$PY -m ML.research.queue --study demo --subjects YES JM EG --protocols S1toS2 S2toS1 --configs FINAL --seeds 0 1 2 --workers 9 --threads 1
# tables + statistics, then the trial plots against the shared truth
$PY -m ML.research.analyze --study demo
$PY -m ML.research.figures --study demo --only trials
```

**What it writes**, in `ML/research/results/demo/<subject>/<direction>/FINAL/`:
- `fast_seed0.json`: settings, the auto-gain decision, passive details, scores per angle;
- `fast_seed0.npz`: all predictions;
- the full-trial plots (`full_trial_inspection_*.png`, seed 0 only).

Each model takes ~5–9 minutes on one CPU core. FINAL has already been run on all 6 subjects in both directions (seed 0): `ML/research/results/final_check/<subject>/<direction>/FINAL/`. Its `details.emg.auto_decision` says "AGC ON" for everyone except CH ("AGC OFF"). Renamed copies of its trial plots are in `notes/final_pipeline/trial_plots/`.

---

## Part 15 — Questions a professor might ask

**Q: What exactly does your model predict?**
The voluntary (active) ankle torque in Nm, at each moment, from the last 200 ms of 4 EMG envelopes and the ankle angle. Not the total measured torque: the passive "rubber band" part is removed (Part 6).

**Q: Is it one model for everyone?**
No, one model per person, trained only on their own first (or second) session. Pre-training on the other 5 people and then fine-tuning (E20) made it worse (−0.03 R²), because people differ too much: angles, strength, even push direction.

**Q: You need the torque sensor to make the training labels. What's the point?**
The sensor is needed once per person, for one training session. After that the model predicts from EMG + angle alone. Our test simulates that with the other session. A new person needs their own training session.

**Q: Why not just predict the measured torque?**
The passive part isn't produced by the muscles, so EMG can't explain it. It depends on angle, session and time. Predicting it would mix "muscle effort" with "tissue stiffness".

**Q: Why an LSTM and not something simpler?**
- We tested a classical method, ridge regression. Its inputs are the EMG envelopes at every 2nd step of the same 200 ms window plus their squares, with one coefficient set per angle.
- It scored 0.62 without auto-gain and 0.86 with it, vs 0.925 for the LSTM models.
- So the LSTM adds something, but auto-gain matters more. The comparison doesn't show which part of the LSTM makes the difference.
- And the original *tiny* LSTM with auto-gain already gets 0.917, so most of the gain comes from data preparation, not model size.

**Q: Why 200 ms windows?**
- That's the project's original choice, and FINAL (= E22) was only ever run with 200 ms.
- A 400 ms window on top of E03 (E29) gave +0.007, mostly at strong angles and not significant after correction. It's what the selection rule picked, but it was dropped because of its labels (it fails on IES), not because of its window.
- E22 with 400 ms was never tested; it's the obvious next step.
- A longer window doesn't delay the output (the answer is still the torque at the newest sample). It just needs more history before the first prediction, and a bit more computation.

**Q: Why test on the other session?**
Because in real use the model is trained once and used later, under changed conditions. Our second session ("retest", recorded later in the same visit) is the closest stand-in we have; testing within the same session gives an over-optimistic score.

**Q: What's the difference between validation and test?**
Validation is used *during* training to decide when to stop (a mock exam). Test is the other session, used to report the score. Honest caveat: we compared many configurations on the test sessions and picked from them, so report 2 adds cross-fitting, leave-one-subject-out and held-out subjects.

**Q: What happens if you remove auto-gain?**
Same small LSTM: 0.826 without → 0.917 with. It's the biggest single improvement, because it removes most of the session-to-session change in EMG size.

**Q: Why not normalise by the MVC (EMG as % of maximum), the textbook method?**
We tested it on top of the report-1 pipeline: 0.887 (÷ MVC peak, E10) and 0.904 (÷ MVC peak at the same angle, E11), vs 0.925 with auto-gain. The MVC was recorded only once, so a fixed MVC scale can't follow the EMG change between the sessions.

**Q: Can auto-gain hide a real change in effort?**
Yes. It can't tell "electrodes give 2× more signal" from "the person pushes 2× harder" (Part 5). It keeps the ups and downs within the last ~45 s but loses the overall level. That works here because both sessions had the same task.

**Q: What happens if you remove the rest-anchored labels?**
On YES/JM/EG: the same R², but more fake torque while relaxed (0.25 vs 0.17 Nm). On IES, which has no passive recordings, the toes-down angles collapse (R² ≈ 0.05 instead of 0.98).

**Q: What happens if you remove the per-angle experts?**
We never removed only the experts. The closest test (E31) also shrinks the LSTM and changes the loss balancing. It costs about −0.01 R² on YES/JM/EG, more on held-out HM (0.895 vs 0.930), and gives occasional big overshoots at toes-down angles.

**Q: Why scale the targets per angle?**
With plain MSE, the strong angles weigh ~550× more than the weakest one (in squared Nm), so the model would ignore the toes-down angles, which were exactly the problem.

**Q: Is there data leakage?** *(leakage = information from the test data sneaking into training)*
- The test session is never used for training or for the training labels.
- The fixed EMG statistics all come from the training session only: the training maximum, the auto-gain priors, and the auto rule's on/off decision.
- When auto-gain is on, it does adapt to the test EMG, but only to *past* samples of the trial being predicted, and never to the torque, exactly as a live device would.
- The test torque is used only by the grader (the reference truth), never by the model.
- An independent code review checked this, and its fixes were applied before the main runs.
- Honest caveats:
  1. The envelope filters are zero-phase, so each EMG value includes some *future* signal (up to ~0.25 s). A live version would need one-directional filters, which add ~0.1–0.2 s of delay.
  2. The training labels use the same construction as the grader.
  3. We picked the best configuration using the test sessions (see above).

**Q: Why is R² only 0.34 at YES −0.486 (S1→S2)?**
- The person only pushes ~1 Nm there, so a typical error of 0.3 Nm caps R². The old error was 1.5 Nm (R² −14).
- The 3 seeds differ a lot at this angle (0.14 to 0.47).
- E03 does a bit better at this one cell (0.47), though both tie over the toes-down angles of all subjects.

**Q: Did you pick the best seed?**
No. On the main subjects every setup was trained with 3 seeds and the **average** is reported; the held-out subjects had 1 seed. FINAL's main score varies by only ~0.002–0.02 between seeds, but a single weak angle can vary much more (YES −0.486: 0.14–0.47). That's exactly why we report the mean.

**Q: How do you know the improvement is real and not luck?**
- At the 2 most toes-down angles, FINAL beats the old version in 12 out of 12 cases (3 subjects × 2 directions × 2 angles), by +0.80 R² on average (95% CI +0.47 to +1.16).
- "95% CI" (confidence interval) means: resampling those 12 differences thousands of times, 95% of the averages fall in that range.
- Three people is few, so this describes these people, not everyone.

**Q: Isn't grading with your own "truth" circular?**
Partly, and I say so.
- The shared truth is the most physical definition available (relaxed torque = passive torque), and the same truth is used for every model, including the old one.
- On the main subjects FINAL's R² is no better than E03's (0.925 vs 0.925), so circularity doesn't create the result there.
- On the held-out subjects the CH part of the gap comes from the auto rule, not the labels.
- The IES part (0.98 vs 0.70) does come from the labels: E03's labels keep the passive torque, FINAL's remove it the same way the grader does. I'd still call that a real gain, since torque while fully relaxed isn't muscle effort, but it's partly a question of definition.
- A truth built with a different rest detector would be a good extra check.

**Q: What is the "auto" rule?**
If the training data shows the person pushing both toes-up and toes-down (> 2 Nm each way), auto-gain is switched off. Auto-gain removes the absolute signal size, which is needed to tell which muscle group is pushing. Only CH triggers it.

**Q: Why didn't the passive holds before each trial help?**
They only change the passive estimate in the *training* labels, and the relaxed level during the trial differs from the hold (the tissue relaxes). Measuring it inside the trial (rest-anchored labels) is more direct.

**Q: Could this run in real time?**
The model itself is fast (a few thousand parameters). But:
1. the zero-phase filters would have to be replaced by one-directional ones (adding ~0.1–0.2 s of delay; the accuracy cost is untested);
2. the `ML/research` code doesn't save the trained network or its normalisation constants (Part 16).

---

## Part 16 — Honest weak points of the code

- **Zero-phase filters.** The envelope and downsampling filters use up to ~0.25 s of future signal. That's fine offline, not for a live device.
- **No saved model.** `ML/research` trains, predicts and saves the predictions; it does **not** save the trained network. To deploy, you'd add `model.save(...)` in `train.fit_predict`, and you'd also need to save the numbers used around the network:
  - **input side:** each muscle's training maximum and the auto-gain starting range (`train_max`, `prior_lo`, `prior_hi`, computed in `emg_norm.normalise`; `run.py` currently doesn't write them out);
  - **output side:** the per-angle σ and the angle list, used to turn predictions back into Nm. These are already saved in `train_info`.
  - (The main script `ML/training/train.py` does save its best model, as `best_lstm.keras`.)
- **FINAL lives only in `ML/research`.** The main training script (`ML/training/train.py --recommended`) still runs the report-1 version: E03 with the auto rule, without rest-anchored labels. Merging them is a next step.
- **The answer key is a definition.** It assumes that torque during EMG-detected rest is purely passive. If someone never fully relaxes in a trial, the line is less certain (JM S2 −0.787: only 4% of samples detected as relaxed).
- **Few people, and one session pair each.** 3 main subjects + 3 held-out (1 seed each). The two sessions are blocks of the same visit, not different days. FINAL was recommended after looking at the held-out subjects, and that rests mostly on one subject (IES).
- **Fast training settings.** Batch 64 instead of the original 8. FINAL wasn't re-run with the original settings.

---

## Glossary

| word | simple meaning | in our project |
|---|---|---|
| regression | predicting a number | torque in Nm |
| feature | an input the model gets | 4 EMG envelopes + angle |
| target | the right answer the model should give | active torque (Part 6) |
| EMG / envelope | muscle electrical signal / its smoothed size | Part 3 |
| radian (rad) | angle unit, 1 rad ≈ 57° | −0.486 rad ≈ 28° toes-down |
| median / percentile / σ | middle value / value below which p% fall / typical spread | Part 5 box |
| normalisation | rescaling numbers to comparable sizes | ÷ max, auto-gain (Part 5) |
| window | a short piece of signal used as one example | 20 steps × 5 numbers |
| tensor | an N-dimensional array of numbers | (windows, 20, 5) |
| training / validation / test | practice / mock exam / exam | S1 80% / S1 20% / S2 (and the reverse) |
| overfitting | memorising instead of learning | limited by early stopping, dropout |
| underfitting | not even learning the training data | — |
| parameter / weight | an adjustable number inside the model | 3,665 of them |
| hyperparameter | a setting we choose, not learned | learning rate, batch size, units… |
| LSTM | network that reads sequences step by step with a memory | 2 layers of 16 units |
| activation function | small fixed formula that lets the network model curves | tanh, sigmoid inside the LSTM |
| dropout | randomly switching off numbers during training | 20% |
| mixture of experts | a separate small output per situation | one expert per ankle angle |
| loss | how wrong the model is | MSE on per-angle scaled torque |
| gradient | the direction to adjust each parameter to lower the loss | computed by Keras |
| optimizer | the rule that adjusts the parameters | Nadam |
| learning rate | size of each adjustment | 0.003, halved when stuck |
| batch | windows used for one adjustment | 64 |
| epoch | one pass over all training windows | up to 150; training stops 15 epochs after the best one, and the best epoch's parameters are kept |
| early stopping | stop when validation stops improving | patience 15 |
| seed | number fixing the randomness | 0, 1, 2 on the main subjects, averaged |
| R² | how much better than guessing the average (1 = perfect) | per angle, floored at −1 |
| RMSE | typical error size, in Nm | per angle |
| phantom torque | prediction while relaxed (should be 0) | mean \|prediction\| at rest |
| confidence interval (CI) | range the true average is likely in | 95% CI from resampling |
| data leakage | test information sneaking into training | avoided (Part 15), with caveats |
