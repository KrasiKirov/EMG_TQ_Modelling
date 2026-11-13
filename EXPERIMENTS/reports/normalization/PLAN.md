# Normalization study: plan and selection rule

Written on 2 November 2026, **before any LSTM run of this study existed**. Its git commit is the proof of that.
Nothing in this file may be changed after the results come in; changes go in a dated note at the end.

## The question

Auto-gain made the toes-down positions much better, but we cannot fully explain it yet, so it is switched off.
The problem it was fixing is still there: for the same push, a muscle's EMG can come out smaller or larger in
session 2 than in session 1 (JM at −0.36 rad: GM and soleus at about a third of their session-1 size).
**Can a simple, explainable way of scaling the EMG recover part of what auto-gain gave?**

## Fixed settings (every run)

| Setting | Value |
|---|---|
| Target | **total measured torque**, nothing subtracted (`passive: none`) |
| Balanced positions | yes: each position's target is divided by its standard deviation in the training part, predictions multiplied back (`balance: scale`) |
| Passive table | not used: with total torque nothing is subtracted, so the clean table plays no part in training or scoring |
| Window | 0.2 s (20 samples at 100 Hz) |
| Auto-gain | **off** |
| Network | LSTM, 2 layers of 8 cells, dropout 0.3 |
| Training | original settings: Nadam 0.003, batch 8, at most 200 epochs, learning rate × 0.7 after 10 flat epochs, early stopping after 25 |
| Direction | learn session 1 → predict session 2 only |
| Seeds | 0, 1, 2 |

## Arms

| Arm | EMG scaling |
|---|---|
| N0 (control) | each muscle ÷ its largest value in session 1 (one sample, 1 kHz envelope) |
| N1 | each muscle ÷ its 99th percentile in session 1's training part |
| N2 | each muscle ÷ its peak in the maximum-effort (MVC) trials |
| N3 | each session ÷ its own 99th percentile per muscle (session 2: all its active-trial EMG, no torque; transductive) |

Reference predictors, scored next to the arms and never chosen: session 1's mean torque at the same position, and a
per-position ridge regression with each arm's scaling (`NR0`–`NR3`, deterministic).

## Stage 1: choose (YES, JM, EG)

Toes-down = the 2 most toes-down positions of each person (6 cells). Other = the remaining positions.

How a group number is made: for each person and position, the median over the 3 seeds; then the median over the cells
of the group. MAE in newton metres is the deciding metric. R² is reported but does not decide, because at toes-down it
swings with tiny offsets.

An arm **passes** if all of these hold, compared with N0:

1. **Better at toes-down:** its toes-down median MAE is lower by at least 0.05 Nm **and** by at least 10 %.
2. **Not just one person:** at least 2 of the 3 people have a lower toes-down MAE (each person's median over their 2 cells).
3. **No damage elsewhere:** its median MAE at the other positions is not higher by more than 0.05 Nm, and no person's
   other-position MAE is higher by more than 10 %.
4. **Not more seed-dependent:** at the toes-down cells, the median of the worst seed's MAE is not higher than N0's.

If several arms pass, the lowest toes-down median MAE wins. If two are within 0.02 Nm of each other, the simpler one
wins, in the order N1, N2, N3 (N3 last because it needs the whole of session 2 before it can predict).
**If no arm passes, N0 stays.**

## Stage 2: confirm (IES, CH, HM)

The winner and N0 are run on the three people who never influenced the choice, same seeds and settings. If N0 is the
winner, N0 and the arm with the lowest Stage-1 toes-down MAE are run, the latter for information only.
**The Stage-2 numbers are the ones to quote.** No choice is made from them.

## Known limits, stated in advance

- 3 seeds: the "5–95 %" range of 3 values is in practice the minimum and maximum.
- One session pair: between-session variability cannot be measured; a third session would be needed.
- Stage 1 still chooses using session 2 of YES, JM and EG; Stage 2 is the clean check.
- Disclosure: before this rule was written, the ridge references (`NR0`–`NR3`) had been run as a pipeline check, and
  EG's per-position ridge scores for NR0 and NR3 were looked at. No LSTM result had been seen.
