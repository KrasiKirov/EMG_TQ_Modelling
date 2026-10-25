# Everything we tried, in plain English

Numbers come from `results_tables/all_results.xlsx` (sheet *Summary*), always for the **main 3 people (YES, JM, EG), both directions**.
- **average R²**: over all 8 angles; each angle is averaged over runs, then floored at −1.
- **toes-down**: the 2 most toes-down angles of each person.
- **error**: RMSE in Nm.

Every result is given against two answer keys:
- **Supervisor key**: measured torque − passive torque from the passive trials of the session being tested. This is the one we use from now on.
- **In-trial key**: measured torque − the relaxed level inside the same trial (what report 2 used).

**Important:** each version scores best on the key that is built the same way as its own training answers.
- Versions trained with the in-trial passive line do well on the in-trial key, but only about 0.46 at the toes-down angles on the supervisor key.
- Versions trained with the passive trials (section 6) do well on the supervisor key (toes-down about 0.72), but only about 0.48 on the in-trial key.
- The reason: the two keys disagree by about 0.6 Nm typically, up to 2 Nm at the toes-down angles, where the pushes are only about 1 Nm. So which version looks best at the toes-down angles depends on which key you believe, and that is exactly the question for the supervisor.

---

## 0. Starting point: the old pipeline (last semester)
1. **Clean the EMG:** keep 30–300 Hz (where muscle signal lives), take the absolute value, and smooth with a 2 Hz filter. This gives a smooth "how active is the muscle" curve.
2. **Scale the EMG:** divide each muscle by its biggest value in the training session, so it runs roughly 0 to 1.
3. **Make the right answer:** voluntary torque = measured − passive. The passive value is one number per angle from the passive trials, read off an automatic table. That table had a bug for JM (+11.7 Nm at −0.07 rad).
4. **Model:** a small LSTM network: 8 memory cells × 2 layers, 1,001 adjustable numbers. It reads the last 0.2 s of EMG + angle.
5. **Fit:** least squares on torque in Nm.

Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.64, toes-down R² 0.37, toes-down error 0.91 Nm. **In-trial key** (report 2): average R² 0.54, toes-down R² 0.04, toes-down error 1.27 Nm.

## 1. Passive-torque ideas (what we subtract)
### Clean passive table (static holds only)
- **What / how:** Use only the part of each passive trial where the foot is still and the torque is steady: angle wobble below 0.005 rad, torque spread below 1 Nm, at least 5 s. In practice that is the first ~20 s. Average it, one value per angle.
- **Why:** Removes JM’s bad table entry. The automatic rule had accepted pieces that were not relaxed.
- **Verdict:** Used inside the next versions (with a correction). The "passive trials only" versions without any correction are running now; see section 6.

### Passive correction α·P + β (`E02_minimal`)
- **What / how:** Inside the training session’s pushing trials, find the relaxed moments from the EMG and measure the torque there. Fit a straight line: relaxed torque ≈ α × (passive-trial value) + β. α came out at about 0.8 for most people. Then subtract α·P + β instead of P.
- **Why:** The relaxed torque inside a pushing trial is only about 80 % of the passive-trial value.
- **Result:** Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.69, toes-down R² 0.24, toes-down error 1.13 Nm. **In-trial key** (report 2): average R² 0.83, toes-down R² 0.64, toes-down error 0.76 Nm.
- **Verdict:** Shown here with the old small network + equal angle weights (E02). The first-report pipeline (E03, section 3) also uses it.

### Passive from the relaxed hold right before each trial (`E04_final_holdgain`)
- **What / how:** Instead of the table, use the mean of the passive-trial hold recorded just before this pushing trial (plus the correction).
- **Why:** That hold is the closest passive measurement in time.
- **Result:** Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.80, toes-down R² 0.42, toes-down error 0.90 Nm. **In-trial key** (report 2): average R² 0.92, toes-down R² 0.82, toes-down error 0.45 Nm.
- **Verdict:** Tie with E03.

### Passive value that relaxes during the trial (drift model) (`E06_final_drift`)
- **What / how:** Passive = α·P + β + γ·P·ln(1 + t/τ): starts at the hold value and slowly drops during the 90 s. τ is chosen on the training data.
- **Why:** The relaxed level slides down during a trial (stress relaxation).
- **Result:** Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.80, toes-down R² 0.42, toes-down error 0.90 Nm. **In-trial key** (report 2): average R² 0.92, toes-down R² 0.83, toes-down error 0.44 Nm.
- **Verdict:** Tie with E03.

### Passive line through the relaxed moments of the same trial ("in-trial") (`E22_restanchor_final`)
- **What / how:** From the EMG, find the moments when all 4 muscles are off. Take the median torque of each relaxed stretch and draw a smooth line through them (about 8 s smoothing). Subtract that line.
- **Why:** Matches the relaxed level at the right moment and follows the drift. It needs no passive trials (IES has none).
- **Result:** Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.80, toes-down R² 0.46, toes-down error 0.92 Nm. **In-trial key** (report 2): average R² 0.92, toes-down R² 0.84, toes-down error 0.45 Nm.
- **Verdict:** Our report-2 choice. The supervisor prefers passive trials, so from now on it is only a comparison.

### Removing the relaxed periods from training (or weighting them ×0.2) (`E24_final_norest`)
- **What / how:** Drop, or count less, the moments when the person is relaxed.
- **Why:** Question: are the relaxed periods just noise?
- **Result:** Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.80, toes-down R² 0.42, toes-down error 0.90 Nm. **In-trial key** (report 2): average R² 0.92, toes-down R² 0.84, toes-down error 0.45 Nm.
- **Verdict:** Same R², but the model then predicts more fake torque while relaxed. Kept them.

## 2. EMG ideas (how we scale the muscle signals)
### Auto-gain (automatic volume control per muscle) (`E19_minimal_agc`)
- **What / how:** Every 0.5 s, for each muscle: floor = its 10th percentile over the past 20 s (its quiet level); top = its 95th percentile over the past 45 s (its strong-push level). Output = (EMG − floor) ÷ (top − floor), kept between 0 and 2. The range is never smaller than 0.35× the training range. It starts from training values and uses only past EMG, never torque. It switches off automatically if the training trials push in both directions (some above +2 Nm, some below −2 Nm). Only CH triggers that.
- **Why:** For the same push, the EMG size changed between sessions (e.g. JM: about half).
- **Result:** Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.79, toes-down R² 0.42, toes-down error 0.92 Nm. **In-trial key** (report 2): average R² 0.92, toes-down R² 0.83, toes-down error 0.47 Nm.
- **Verdict:** Shown here on the old small network (E19). It is also part of E03 and the new pipeline.

### EMG ÷ maximum push (MVC), once for the whole leg (`E10_final_mvcglobal`)
- **What / how:** Divide each muscle by its EMG during the maximum push, recorded once at the start of the visit.
- **Why:** Standard EMG normalisation.
- **Result:** Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.73, toes-down R² 0.32, toes-down error 1.03 Nm. **In-trial key** (report 2): average R² 0.89, toes-down R² 0.81, toes-down error 0.57 Nm.
- **Verdict:** Worse than auto-gain: the MVC is recorded once, but the EMG size changes later.

### EMG ÷ maximum push at that angle (`E11_final_mvcangle`)
- **What / how:** Same, using the maximum push at the same angle.
- **Why:** Muscles produce different EMG at different lengths.
- **Result:** Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.76, toes-down R² 0.39, toes-down error 0.98 Nm. **In-trial key** (report 2): average R² 0.90, toes-down R² 0.83, toes-down error 0.55 Nm.
- **Verdict:** Worse than auto-gain.

### Random EMG-size changes during training (augmentation) (`E16_final_ga50`)
- **What / how:** During training only, multiply each muscle’s EMG in each window by a random factor (typical spread ±50 %), so the model gets used to size changes.
- **Why:** Make the model insensitive to EMG size.
- **Result:** Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.81, toes-down R² 0.45, toes-down error 0.89 Nm. **In-trial key** (report 2): average R² 0.92, toes-down R² 0.83, toes-down error 0.42 Nm.
- **Verdict:** No gain on top of auto-gain (they fix the same problem).

### Each session scaled by its own EMG level (`E26_final_sessp99`)
- **What / how:** Divide each session by its own 99th-percentile EMG.
- **Why:** Simple per-session normalisation.
- **Result:** Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.67, toes-down R² 0.43, toes-down error 0.91 Nm. **In-trial key** (report 2): average R² 0.78, toes-down R² 0.84, toes-down error 0.45 Nm.
- **Verdict:** Needs the whole test session in advance (not allowed); breaks JM.

## 3. Fit ideas (how much each angle counts)
### Divide each angle by its typical swing (= weighted least squares)
- **What it is:** for each angle, the **typical swing** σ is the standard deviation of the voluntary torque at that angle in the training data. It is not the ankle moving; it is how much the torque goes up and down there.
  - YES: σ = 0.64 Nm at −0.49 rad (tiny pushes), 15.0 Nm at +0.14 rad (big pushes).
- **How it is computed:**
  - Before fitting, every target is divided by its angle’s σ (σ is at least 0.1 Nm).
  - After predicting, the output is multiplied back by σ, so it is in Nm again.
  - This is the same as weighted least squares with weight 1/σ² per angle.
- **Why:** plain least squares in Nm lets the strong angles dominate. For YES, the 2 most toes-down angles carried only 1 % of the fit. After dividing, every angle carries 12.5 %.
- **Result:** part of the first-report pipeline below (E03). When we took it out of the new pipeline (1 run), the main 3’s toes-down R² (in-trial key) fell from 0.86 to 0.71.

### The first-report pipeline (`E03_final`): clean passive table × correction + auto-gain + per-angle corrections + divide by swing
- **Result:** Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.80, toes-down R² 0.42, toes-down error 0.91 Nm. **In-trial key** (report 2): average R² 0.93, toes-down R² 0.85, toes-down error 0.44 Nm.

### ×3 extra weight on the 2 most toes-down angles (`E13_final_pf3`)
- **What / how:** On top of the division, count the 2 most toes-down angles 3× more.
- **Result:** Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.80, toes-down R² 0.42, toes-down error 0.92 Nm. **In-trial key** (report 2): average R² 0.92, toes-down R² 0.83, toes-down error 0.45 Nm.
- **Verdict:** Tie.

### ×10 extra weight (`E14_final_pf10`)
- **What / how:** Same with 10×.
- **Result:** Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.79, toes-down R² 0.41, toes-down error 0.94 Nm. **In-trial key** (report 2): average R² 0.92, toes-down R² 0.83, toes-down error 0.46 Nm.
- **Verdict:** Worse; the toes-down angles did not improve.

## 4. Model ideas
### Per-angle corrections ("experts") + 16 memory cells
- **What / how:** Same LSTM type but 16 memory cells × 2 layers. The shared read-out (a weighted sum of the 16 summary numbers) gets one extra set of 16 multipliers per training angle. They start at 0 and are kept small. The ankle angle picks that angle’s set, and between two angles it blends the two. 3,665 numbers in total.
- **Why:** The EMG→torque link changes with muscle length.
- **Verdict:** Small gain (about +0.01), fewer overshoot spikes. Part of E03 and the new pipeline.

### Longer window: 0.4 s instead of 0.2 s (`E29_final_win40`)
- **What / how:** The model looks at the last 0.4 s of EMG instead of 0.2 s.
- **Why:** More history may help.
- **Result:** Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.81, toes-down R² 0.43, toes-down error 0.90 Nm. **In-trial key** (report 2): average R² 0.93, toes-down R² 0.84, toes-down error 0.44 Nm.
- **Verdict:** Small gain overall (mostly strong angles), none at toes-down. Next: try it with the passive-trial pipeline (running now, section 6).

### Pre-train on the other 5 people, then refit on this person (`E20_xsub_agc`)
- **What / how:** First fit the model on everyone else’s data, then continue fitting on this person.
- **Why:** More data.
- **Result:** Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.78, toes-down R² 0.42, toes-down error 0.94 Nm. **In-trial key** (report 2): average R² 0.88, toes-down R² 0.72, toes-down error 0.51 Nm.
- **Verdict:** Worse: people differ too much.

### Straight-line model (ridge regression) (`E27_ridge`)
- **What / how:** No network. A weighted sum of the recent EMG values (and their squares), with one set of weights per angle.
- **Why:** Is a neural network needed at all?
- **Result:** Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.47, toes-down R² 0.07, toes-down error 1.27 Nm. **In-trial key** (report 2): average R² 0.62, toes-down R² 0.54, toes-down error 0.81 Nm.
- **Verdict:** Clearly worse. With auto-gain (E28) much better, but still behind the network.

### Old small network inside the in-trial pipeline (`E31_restanchor_agc_lstm8`)
- **What / how:** In-trial passive line + auto-gain, but the old 8-cell network without per-angle corrections.
- **Why:** Simplest good version?
- **Result:** Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.78, toes-down R² 0.45, toes-down error 0.94 Nm. **In-trial key** (report 2): average R² 0.92, toes-down R² 0.84, toes-down error 0.50 Nm.
- **Verdict:** Slightly behind, with occasional spikes.

## 5. Checks done in our first report (not in the tables)
- **Other network types** (GRU, CNN, plain network, bigger LSTM) on the same data: none better everywhere.
- **Angle-sensor leak test:** we replaced the measured angle (the footplate bends slightly when pushing) with the set angle. No change, so the model does not cheat through the angle.
- **Re-zeroing each test trial with its own torque:** rejected, because it uses the answer we are trying to predict.

## 6. Passive trials only (supervisor’s rule), running now
Training answers use the passive trials of the training session; we score with the passive trials of the tested session. No correction from the relaxed moments.
- **Old pipeline with the clean passive table (JM bug removed)** (`E36_static_old`): Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.78, toes-down R² 0.46, toes-down error 0.88 Nm. **In-trial key** (report 2): average R² 0.69, toes-down R² 0.16, toes-down error 1.23 Nm.
- **Passive trials only + auto-gain + per-angle corrections + divide by swing** (`E37_static_full`): Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.89, toes-down R² 0.72, toes-down error 0.49 Nm. **In-trial key** (report 2): average R² 0.81, toes-down R² 0.48, toes-down error 0.91 Nm.
- **Same with a 0.4 s window** (`E38_static_full_win40`): Main 3 people (3 runs). **Supervisor key** (passive trials of the tested session): average R² 0.91, toes-down R² 0.74, toes-down error 0.46 Nm. **In-trial key** (report 2): average R² 0.82, toes-down R² 0.49, toes-down error 0.89 Nm.

## 7. All versions in one table (main 3 people, both directions)
| version | runs | supervisor key: average R² | supervisor key: toes-down R² | supervisor key: toes-down error (Nm) | in-trial key: average R² | in-trial key: toes-down R² |
|---|---|---|---|---|---|---|
| `E01_old` | 3 | 0.64 | 0.37 | 0.91 | 0.54 | 0.04 |
| `E02_minimal` | 3 | 0.69 | 0.24 | 1.13 | 0.83 | 0.64 |
| `E03_final` | 3 | 0.80 | 0.42 | 0.91 | 0.93 | 0.85 |
| `E04_final_holdgain` | 3 | 0.80 | 0.42 | 0.90 | 0.92 | 0.82 |
| `E06_final_drift` | 3 | 0.80 | 0.42 | 0.90 | 0.92 | 0.83 |
| `E10_final_mvcglobal` | 3 | 0.73 | 0.32 | 1.03 | 0.89 | 0.81 |
| `E11_final_mvcangle` | 3 | 0.76 | 0.39 | 0.98 | 0.90 | 0.83 |
| `E13_final_pf3` | 3 | 0.80 | 0.42 | 0.92 | 0.92 | 0.83 |
| `E14_final_pf10` | 3 | 0.79 | 0.41 | 0.94 | 0.92 | 0.83 |
| `E16_final_ga50` | 3 | 0.81 | 0.45 | 0.89 | 0.92 | 0.83 |
| `E18_minimal_ga50` | 3 | 0.70 | 0.27 | 0.99 | 0.86 | 0.76 |
| `E19_minimal_agc` | 3 | 0.79 | 0.42 | 0.92 | 0.92 | 0.83 |
| `E20_xsub_agc` | 3 | 0.78 | 0.42 | 0.94 | 0.88 | 0.72 |
| `E21_xsub_trainmax` | 3 | 0.66 | 0.23 | 1.16 | 0.82 | 0.67 |
| `E22_restanchor_final` | 3 | 0.80 | 0.46 | 0.92 | 0.92 | 0.84 |
| `E23_restanchor_minimal` | 3 | 0.69 | 0.34 | 1.06 | 0.85 | 0.76 |
| `E24_final_norest` | 3 | 0.80 | 0.42 | 0.90 | 0.92 | 0.84 |
| `E25_final_rest02` | 3 | 0.80 | 0.42 | 0.91 | 0.92 | 0.84 |
| `E26_final_sessp99` | 3 | 0.67 | 0.43 | 0.91 | 0.78 | 0.84 |
| `E27_ridge` | 3 | 0.47 | 0.07 | 1.27 | 0.62 | 0.54 |
| `E28_ridge_agc` | 3 | 0.72 | 0.32 | 1.10 | 0.86 | 0.80 |
| `E29_final_win40` | 3 | 0.81 | 0.43 | 0.90 | 0.93 | 0.84 |
| `E31_restanchor_agc_lstm8` | 3 | 0.78 | 0.45 | 0.94 | 0.92 | 0.84 |
| `E32_restanchor_noagc_moe` | 1 | 0.71 | 0.39 | 0.99 | 0.88 | 0.84 |
| `E33_final_noscale` | 1 | 0.80 | 0.46 | 0.88 | 0.89 | 0.70 |
| `E34_final_oldpassive` | 1 | 0.81 | 0.71 | 0.51 | 0.72 | 0.48 |
| `E35_final_lstm8` | 1 | 0.80 | 0.49 | 0.87 | 0.92 | 0.85 |
| `E36_static_old` | 3 | 0.78 | 0.46 | 0.88 | 0.69 | 0.16 |
| `E37_static_full` | 3 | 0.89 | 0.72 | 0.49 | 0.81 | 0.48 |
| `E38_static_full_win40` | 3 | 0.91 | 0.74 | 0.46 | 0.82 | 0.49 |
