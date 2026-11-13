# Likely questions, and where the deck answers them

"S#" = slide number. Slides 15–18 are backup slides: keep them after the talk and open one if someone asks.

## The experiment and the data

**1. What exactly is recorded?** (S2)
- 4 EMG signals: calf GM, GL, SOL and shin TA.
- The ankle angle and the torque, recorded at 1000 Hz and used every 10 ms.
- 6 people, 2 sessions in one visit, 8 angles per session, ~90 s of push / relax per angle.

**2. Were people only asked to push, or also to pull?** (S2)
- The trial comments never mention pulling; IES's comments say "PUSH". The torque data answers the rest.
- YES, JM, EG, IES and HM only push the toes down.
- CH pulled the toes up (+10 to +30 Nm, shin muscle) at its most toes-down angles.
- At one angle (−0.31 / −0.33 rad) CH pulled in session 1 and pushed in session 2.
- Whether pulling was the instruction for CH can't be told from the files; ask the lab.

**3. Are "session 1" and "session 2" different days?** (S2, S13)
- No. "Test" and "retest" are in the same visit and the same file.
- A different-day test is the main thing still missing.

**4. Why are the toes-down angles the hard ones?** (S2, S3)
- With the toes down the calf is short and weak. The person can only make ~1 Nm there, against ±15 Nm at strong angles.
- A small error in Nm is therefore a big error relative to the signal.

**5. Why one model per person, not one for everyone?** (S2, S12)
- Pre-training on the other 5 people made results worse (0.885 vs 0.925).
- People differ too much in angles, strength, and even push direction (CH).

## Passive and voluntary torque

**6. What is passive torque, and why remove it?** (S4)
- The sensor measures everything, including the stretched ankle pulling back like a rubber band.
- We want only what the muscles do: voluntary = measured − passive.

**7. How was passive torque computed last semester?** (S4, S5)
- From separate relaxed recordings (the passive trials): the mean torque of each steady part.
- That gives one number per angle, interpolated between angles and subtracted from every sample.
- For JM, a movement segment slipped into that table: a 12 Nm error at −0.078 rad.

**8. How is it computed now?** (S4, S15)
1. EMG only: average activity of the 4 muscles, as a fraction of each muscle's session range.
2. "Relaxed" starts below 0.03 and ends above 0.06, for at least 0.5 s.
3. Take the median torque of each relaxed stretch.
4. Draw a smooth line through those medians (σ = 8 s). That line is the passive torque at every moment.
5. Voluntary = measured − line.

**9. Why is the new passive value lower than the old one?** (S4)
- For YES at −0.486 rad: separate recording 10.6 Nm, real relaxed level 8.4–9.0 Nm, drifting down during the trial.
- Passive torque relaxes over time and depends on what the ankle just did (Duong 2001; Kay & Blazevich 2009). A recording made at another moment doesn't match.

**10. Isn't it circular to find "relaxed" from EMG and then predict torque from EMG?** (S4, S13, S15)
- The EMG only says *when* the person is relaxed. The *value* at those moments is the measured torque.
- The line is very smooth (8 s), so it cannot copy the pushes.
- The model never sees the line; it only defines the right answer.

**11. Doesn't the answer key favour FINAL, since FINAL's labels use the same idea?** (S13)
- Partly, and the deck says so.
- Check: report 1's pipeline, whose labels are built differently (passive recordings + a fitted correction), scores the same 0.925 on that key.

**12. What if the person is never relaxed?** (S13, S15)
- Every trial had at least 2 relaxed stretches (median 15, ~24 % of the time).
- The thinnest was JM session 2 at −0.50 rad: relaxed only 0.9 % of the time, so the line is less certain there.

**13. Does the model need the passive line when it predicts?** (S4, S7)
- No. It predicts voluntary torque directly from EMG + angle. The line is only used to make training labels and the answer key.

## EMG processing

**14. What happens to the raw EMG?** (S6)
- 30–300 Hz filter, absolute value, 2 Hz smoothing, downsampled to 100 Hz.
- Divided by the training-session maximum, then auto-gain.

**15. What is auto-gain, and does it cheat?** (S5, S6, S16)
- Per muscle: (EMG − floor) ÷ (top − floor).
  - floor = 10th percentile of the past 20 s; top = 95th percentile of the past 45 s.
  - It starts from the training-session values.
- It uses only past EMG: no torque, nothing from the future. So it could run live.
- JM example: the same push gives a calf signal 0.40× as big in session 2; after auto-gain, 0.74×. Most of the gap is closed, not all.

**16. Why is auto-gain off for CH?** (S6, S16)
- Rule: if some training trials push above +2 Nm and others below −2 Nm, auto-gain goes off. Only CH triggers it.
- Rescaling each muscle to its own range hides which muscle (calf or shin) is working.
- At CH's flipped angle: R² −3.3 / −5.5 with auto-gain, 0.98 / 0.95 without.

**17. Why not normalise by the MVC (maximum effort)?** (S12, S16)
- It scored worse: 0.887 / 0.904 vs 0.925.
- The MVC was recorded once, but the EMG size changes between sessions.

## The model and training

**18. Are we still using an LSTM? What is "mixture of experts"?** (S7)
- Yes: 2 LSTM layers of 16.
- On top of them, one small "expert" correction per training angle, picked (and blended) by the ankle angle.
- 3,665 numbers in total, vs 1,001 before.

**19. Why not one separate model per angle?** (S7)
- The experts are that idea. All angles share the LSTM, so the weak angles still learn from the strong ones.

**20. How is it trained, on how much data, and how long does it take?** (S7, S18)
- ~57,000 windows of 200 ms: the first 80 % of each training trial. The last 20 % is validation.
- Optimiser Nadam, batch 64. Training stops 15 epochs after validation last improved (17–73 epochs).
- ~5 min per model on a laptop CPU.

**21. How do you know it isn't memorising?** (S7)
- 3,665 numbers vs ~57,000 windows, 20 % dropout and early stopping.
- It is tested on a session it never trained on, in both directions.

**22. Do we even need deep learning?** (S12)
- A linear model (ridge) gets 0.624, or 0.863 with auto-gain. The LSTM gets 0.925.
- But most of the gain is the data: the old tiny LSTM with the new data steps gets 0.917.

**23. Why 200 ms windows?** (S12)
- 400 ms adds +0.007, only at the strong angles. It's a next step to try with FINAL.

## Checking

**24. What does R² mean, and how can it be −14?** (S3, S18)
- R² = 1 − squared error ÷ variance of the truth. 1 = perfect, 0 = guessing the average, < 0 = worse.
- At YES −0.486 the torque only varies by ~0.4 Nm, so a 1.5 Nm offset gives −14. That's why the error in Nm is always shown too.

**25. What was wrong with how the model was checked before?** (S8)
- Each pipeline was graded against its own answer key, in one direction, with one run, and with R² only.
- A wrong passive value shifted the answer key and the prediction together, so the error was invisible.
- With the shared key the old model drops from 0.68 to 0.54.

**26. Could the results be luck from one training run?** (S8, S17)
- 3 runs, averaged, never the best one. For FINAL the run-to-run spread (std of R²) is 0.002–0.020 per subject and direction.
- Paired test: FINAL beats old in 12 / 12 toes-down results (p = 0.0005) and 46 / 48 overall.

**27. Did fixing the toes-down angles make other angles worse?** (S9, S17)
- Main subjects: better at 46 / 48 angles. The other 2 drop by less than 0.01.
- Extra people: better at 42 / 48. The drops are at middle / toes-up angles, never toes-down; the largest is 0.92 → 0.83.

**28. Does it work on people who weren't used to design it?** (S9, S13, S17)
- On IES, CH and HM (1 run each), toes-down R² goes 0.70 → 0.97.
- Caveat: FINAL was picked after seeing them, so they are not fully independent proof.

**29. How was the winner chosen?** (S8, S13)
- A written rule (best score on YES / JM / EG) picked the 400 ms variant.
- FINAL was recommended after the extra people showed the variant breaks for IES, who has no passive recordings. The deck states this openly.

**30. Does the model invent torque while the person is relaxed?** (S9)
- Much less: 1.25 Nm → 0.17 Nm on average.

## Limits and practical use

**31. Can this run in real time?** (S13)
- Not yet. The EMG filters use ~0.25 s of future signal.
- Auto-gain and the model are already causal; the filters need a one-directional version.

**32. What is still not good?** (S11, S13)
- YES −0.486: R² 0.34, ~0.3 Nm error on a ~1 Nm push.
- Also: only 6 people, same-visit sessions, and the answer key's assumption that quiet EMG means zero voluntary torque.

**33. What else was tried?** (S12)
- 14 configurations shown (23 tested): holds, a drift model, MVC scaling, extra weight on toes-down angles, augmentation, pre-training, ridge, and a 400 ms window. None clearly beat FINAL.

**34. Where are the code and the details?** (S6, S14)
- Code: `ML/research`, run with `--config FINAL`.
- Details: `notes/report2/round2_report.md` and `notes/final_pipeline/final_code_pipeline.md`.
- All full-trial plots: `notes/final_pipeline/trial_plots/`.
