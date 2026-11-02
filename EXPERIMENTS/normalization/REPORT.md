# Normalization study: can a simple EMG scale do what auto-gain did?

*All runs on 2 November 2026. Plan and selection rule fixed before any LSTM result: `PLAN.md`, commit 921ecdd.*

## The short version

**No simple EMG scaling beat the control. N0, dividing each muscle by its session-1 maximum, stays.** That was decided
on YES, JM and EG by a rule written before any result, and IES, CH and HM, who never influenced anything, confirmed
it.

The numbers to quote, from IES, CH and HM, trained on session 1 and tested on session 2, total torque:

| N0, the model we keep | MAE | typical peak | R² | VAF |
|---|---|---|---|---|
| Toes-down positions | **0.86 Nm** | 6.4 Nm | 0.86 | 97 % |
| All other positions | **0.76 Nm** | 20.0 Nm | 0.98 | 98 % |

Four things we learned:

1. **N1 (99th percentile) and N2 (maximum-effort trial) cannot fix a change between sessions.** Any fixed number per
   muscle is absorbed by the model: a straight-line model gives the same predictions, within 0.003 Nm, with N0, N1 or
   N2.
2. **N3 (each session's own scale) helps some people a lot and hurts others.** It helped EG and IES at toes-down and
   hurt JM, CH and HM. One number per muscle per session cannot follow a change that differs from angle to angle.
   Auto-gain can, because its scale keeps moving.
3. **Most of the toes-down error is an offset, not a wrong shape.** The prediction follows the ups and downs well (VAF
   89 to 97 %) but often sits at the wrong level, which is why R² is lower. For YES, the resting torque itself moved
   by about 2.5 Nm between sessions. No EMG scaling can see that.
4. **Correcting the level with the relaxed holds is not a general fix.** Shifting each position by the change in the
   30-second relaxed hold recorded before every trial fixed YES (2.37 to 0.48 Nm). But on CH and HM, the people it was
   checked on afterwards, it made CH worse and only helped HM. It was an idea from after Stage 1, so we report it as
   exploratory and do not use it.

The LSTM beat both reference predictors everywhere: a per-position ridge model, and a flat line at session 1's average.

## Why we ran this

Auto-gain made the toes-down positions much better. But it is hard to explain, so for now it is switched off. That
leaves one problem from the video without a fix: **the same push can give a smaller, or larger, EMG in session 2 than
in session 1**, and dividing session 2 by session 1's numbers cannot correct that. For JM at minus 0.36 radians, the
gastrocnemius medialis and the soleus came out at only about a third of their session-1 size. When a muscle reads
weaker, the model predicts a smaller push.

So the question is simple. **Is there a plain, one-number-per-muscle way of scaling the EMG that does part of
auto-gain's job?**

## What stayed the same in every run

| Setting | Value |
|---|---|
| What the model predicts | the **total** measured torque. Nothing is subtracted. |
| Balanced positions | each position's target is divided by how much the torque swings there (its standard deviation in session 1's training part). Predictions are multiplied back. |
| Passive table | not used. With total torque there is nothing to subtract, so the clean table plays no part. |
| Window | 0.2 seconds: 20 samples at 100 per second, of the four muscle envelopes plus the angle |
| Auto-gain | off |
| Network | the same small LSTM: 2 layers of 8 cells, dropout 0.3 |
| Training | the original settings: batch 8, at most 200 epochs, early stopping after 25 epochs without improvement |
| Direction | learn session 1, predict session 2 |
| Seeds | 0, 1 and 2 |
| Hardware | Ryzen 7 3700X CPU (TensorFlow 2.21, CPU only), 1 thread per run, 8 runs at a time in Stage 1 and 9 in Stage 2 |

Two consequences of predicting the total torque are worth knowing before reading the numbers.

First, **the numbers are not directly comparable with D01 or the baseline**. Those predicted the active torque, measured
minus passive. Here the answer key includes the passive part, so each position's average is further from zero, and the
model must also get that level right. The MAE in newton metres is still a fair way to compare the four arms of this
study, because they all share the same answer key.

Second, **balanced positions now divides the whole torque, passive part included**. At a toes-down position where the
torque swings by only one newton metre, a passive level of a few newton metres becomes a large number for the network
to output. That is how the setting was defined, so we kept it, but it is a difference from D01.

## The four ways of scaling the EMG

For each one: what it does, why it might help, and could it be cheating? By cheating we mean: does information from
session 2's torque sneak into the model?

**N0, the control: session 1's maximum.** Each muscle is divided by the single largest value it reached anywhere in
session 1's eight active trials. Session 2 is divided by the same numbers. This is what the baseline does. As the video
showed, two things are fragile here: one spike can set a whole muscle's scale, and for some people that maximum falls in
the validation part. Not cheating.

**N1: session 1's 99th percentile.** Each muscle is divided by the level it stays below 99 percent of the time, over
session 1's training part only. A single spike can no longer set the scale, and the validation part is no longer
touched. But session 2 is still divided by session 1's numbers, so it does nothing about the change between sessions.
Not cheating.

**N2: the maximum-effort trial (MVC).** Each muscle is divided by its peak during the maximum-effort trials, so the EMG
becomes a percentage of maximal activation. This is the textbook answer, and the one a supervisor will ask about. The
catch: our MVC trials were recorded once, at the start of the visit, so session 2 is again divided by a number from
before session 2. Not cheating.

**N3: each session's own 99th percentile.** Each session is divided by its own 99th percentile per muscle. For session
1 that comes from its training part. For session 2 it comes from all of session 2's active-trial EMG, without any
torque. This is the static version of auto-gain: one number per muscle per session, instead of one that moves every
half second. It uses no session-2 torque, so it is not cheating in that sense. But it needs the whole of session 2's EMG
before it can predict anything, so a real device could not do it as is: it would need a short calibration block first.
We call this *transductive*.

Next to the four LSTM arms, two **reference predictors** are scored, and never chosen:

- **Session 1's average at that position**: for each position, predict a flat line at the average torque of session
  1's trial at the same angle. Any useful model must beat this.
- **Ridge regression**, one straight-line model per position, with each arm's scaling (`NR0` to `NR3`). It has no
  seeds, so it shows the effect of the scaling without any training luck.

## How the winner is chosen

The rule was written before any LSTM result existed. In short: an arm must beat N0's toes-down MAE by at least 0.05
newton metres **and** 10 percent, for at least 2 of the 3 people, without making the other positions worse by more than
0.05 newton metres or any person's other positions worse by more than 10 percent, and without a worse worst seed.
If none passes, N0 stays. The full rule is in `PLAN.md`.

Every group number is made the same way: first the median over the 3 seeds for each person and position, then the
median over the cells of the group. Toes-down means each person's 2 most toes-down positions. R² is shown, with values
below zero displayed as zero, but it does not decide: at toes-down it swings with tiny offsets.

The choice is made on YES, JM and EG (Stage 1). The winner and N0 are then run on IES, CH and HM, who never influenced
anything (Stage 2). **The Stage 2 numbers are the ones to quote.**

## Stage 1: what happened on YES, JM and EG

### The scoreboard

Every number is an MAE in newton metres, the typical size of the miss. Next to it is the typical peak torque of those
positions, so you can judge how big the miss is. R² is in brackets, with values below zero shown as zero.

| | toes-down MAE (peak 5.2 Nm) | other positions MAE (peak 18.9 Nm) |
|---|---|---|
| **N0** session-1 maximum (control) | **0.85** (R² 0.47) | **1.14** (R² 0.95) |
| N1 session-1 99th percentile | 0.99 (R² 0.50) | 1.02 (R² 0.94) |
| N2 maximum-effort trial | 0.97 (R² 0.51) | 1.28 (R² 0.93) |
| N3 each session's own 99th percentile | **0.71** (R² 0.36) | 1.41 (R² 0.94) |
| Ridge, scaling N0, N1 or N2 (identical, see below) | 1.29 (R² 0.23) | 1.73 (R² 0.87) |
| Ridge, scaling N3 | 0.80 (R² 0) | 1.78 (R² 0.91) |
| Session 1's average at that position | 2.13 (R² 0) | 6.86 (R² 0) |

The same MAEs per person:

| | EG toes-down | JM toes-down | YES toes-down | EG other | JM other | YES other |
|---|---|---|---|---|---|---|
| typical peak | 5.5 | 2.0 | 8.0 | 26.5 | 6.1 | 26.4 |
| **N0** | 0.85 | 0.43 | 2.37 | 1.21 | 0.79 | 1.45 |
| N1 | 0.99 | 0.43 | 2.43 | 0.98 | 0.74 | 1.93 |
| N2 | 0.97 | 0.42 | 2.43 | 1.16 | 0.62 | 2.08 |
| N3 | **0.52** | 0.56 | 2.30 | 1.25 | **1.27** | 1.75 |
| session-1 average | 2.79 | 0.66 | 2.36 | 9.81 | 1.64 | 10.80 |

How much luck is in these numbers? Across the three seeds, the best and the worst run of the same person and position
usually differ by about 0.1 Nm at toes-down and 0.2 to 0.3 Nm elsewhere. With only 3 seeds, a 5 to 95 percent range
would mean nothing, so the dashboard shows each seed instead.

### The rule's verdict: no arm passes, N0 stays

| Arm | toes-down vs N0 | people better at toes-down | other positions vs N0 | worst person elsewhere | verdict |
|---|---|---|---|---|---|
| N1 | 0.14 Nm worse | 1 of 3 | 0.12 Nm better | 33 % worse (YES) | fails |
| N2 | 0.12 Nm worse | 1 of 3 | 0.14 Nm worse | 43 % worse (YES) | fails |
| N3 | **0.14 Nm better (17 %)** | 2 of 3 (EG, YES) | 0.27 Nm worse | **62 % worse (JM)** | fails only the "do no harm elsewhere" test |

N3 cleared the toes-down bar, the two-people bar and the worst-seed bar. It failed because it damaged the other
positions, mostly for JM. As the plan says, when no arm passes, N0 stays, and Stage 2 runs N0 next to the arm with the
lowest toes-down MAE, N3, for information only. The full mechanical output is in `decision_stage1.md`.

### Finding 1: N1 and N2 cannot help, and the ridge model shows why

Look at the ridge rows. With scalings N0, N1 and N2, ridge gives **the same predictions: they never differ by more than 0.003 Nm**.
That is not a coincidence. Each of these three scalings multiplies a muscle by one fixed number, the same number in
both sessions. A straight-line model simply absorbs that number into its weight: multiply the input by 3 and the
weight comes out 3 times smaller. No information is added or removed.

The LSTM is not a straight line, so it does react to the input size: a different size changes where its curves
saturate and how training moves. That is why N1 and N2 give slightly different numbers from N0. But the difference is
about training, not information. Both came out a little worse at toes-down (by 0.12 to 0.14 Nm), which is only a bit
more than the seed-to-seed wobble. **The lesson: a fixed per-muscle scale, whichever one, cannot fix a change between
sessions.** Only something that looks at session 2 itself can.

That does not make N1 or N2 useless in general. N1 is the more robust choice if one spike ever sets a muscle's maximum,
and N2 is the one that makes the EMG a percentage of maximal effort, which a supervisor may want for interpretation.
They just do not touch the problem this study was about.

### Finding 2: N3 helps where the session change is uniform, and hurts where it is not

N3 is the only arm that changes the information, because session 2 gets its own scale. How big were those scales? Here
is session 2's 99th percentile divided by session 1's, per muscle, over all active trials:

| | GM | GL | SOL | TA |
|---|---|---|---|---|
| YES | 1.03 | 1.03 | 1.21 | 1.35 |
| JM | 0.79 | 1.02 | 0.71 | 0.93 |
| EG | 1.64 | 1.37 | 1.11 | 0.98 |

So for EG the gastrocnemius medialis reads 64 percent larger in session 2. N3 shrinks it back, and EG's toes-down error
falls from 0.85 to 0.52 Nm, the biggest single improvement in the study. For JM the gastrocnemius medialis and the
soleus read smaller in session 2, so N3 scales them up.

The problem: **one number per muscle per session is set by the strongest pushes**, and those happen at the neutral and
toes-up positions. If the change between sessions were the same at every angle, one number would fix everything. It
is not. For JM, after N3 the neutral and toes-up positions predict too much push: at plus 0.20 radians the average
miss moves from 0.4 Nm too little to 1.5 Nm too much. For EG, the opposite happens at the neutral positions. So N3
fixes some angles and breaks others.

This is exactly the job auto-gain did, and why it worked better: its gain moves every half second, so it can be right
at one angle and also right at another. A single number per session cannot. **So a plain, one-number-per-muscle
scaling does part of auto-gain's job only for people whose change between sessions is the same at all angles.**

N3 also has the practical catch described above: it needs session 2's EMG before predicting. Even if it had passed,
a real device would need a short calibration block at the start of every session.

### Finding 3: YES's toes-down error is not about EMG at all

At YES's two toes-down positions, every predictor misses by about 2.3 Nm in the same direction, all four LSTM arms,
both ridge models, **and the session-1 average**. At minus 0.49 radians, session 2's torque averages plus 8.2 Nm, while
session 1's averaged about 5.8 Nm at the same angle. The models say what session 1 said.

At these positions the foot is far toes-down, the muscles can add only a little, and the
torque is mostly the passive pull of the stretched tissues. That passive level moved by about 2 Nm between the
sessions. No scaling of the EMG can see that, because it is not in the EMG. This one person decides much of the
toes-down median: it is the largest error in Stage 1.

You said a passive level as an input is something to keep in mind for later. This is the case where it would matter.

### Finding 4: the reference predictors

- The LSTM is clearly better than **session 1's average at that position**: 0.85 against 2.13 Nm at toes-down, and
  1.14 against 6.86 Nm elsewhere. The one exception is YES at toes-down, where the LSTM (2.37) is no better than the
  flat line (2.36), for the reason in Finding 3.
- The LSTM is also clearly better than **ridge** with the same scaling: 0.85 against 1.29 Nm at toes-down, and 1.14
  against 1.73 Nm elsewhere. The small network earns its place.
- Ridge reacts to N3 the same way as the LSTM: better at toes-down (1.29 to 0.80), not better elsewhere. So the N3
  effect is not training luck. It comes from the scaling itself.

### A word on R²

At toes-down, R² is low everywhere and jumps around. For YES it is zero for every arm, because the torque hardly moves
inside the trial while the whole trial is 2.3 Nm off. That is why the rule decides on MAE in newton metres. N3's
toes-down R² (0.36) is lower than N0's (0.47) even though its MAE is lower: JM's toes-down positions got worse with
N3, and with so little movement there, a small offset drives R² far down.

## Stage 2: the check on IES, CH and HM

Following the plan, because no arm passed, Stage 2 ran N0 next to the arm with the lowest toes-down error, N3, for
information only. Nothing about these three people was looked at before this point.

| | toes-down MAE (peak 6.4 Nm) | other positions MAE (peak 20.0 Nm) |
|---|---|---|
| **N0** session-1 maximum, the model we keep | **0.86** (R² 0.86, VAF 97 %) | **0.76** (R² 0.98, VAF 98 %) |
| N3 each session's own 99th percentile | 0.70 (R² 0.92) | 0.99 (R² 0.97) |
| Ridge, scaling N0, N1 or N2 (identical again) | 1.08 (R² 0.89) | 1.54 (R² 0.96) |
| Ridge, scaling N3 | 1.45 (R² 0.66) | 2.08 (R² 0.89) |
| Session 1's average at that position | 3.16 (R² 0) | 6.83 (R² 0) |

Per person, MAE in Nm, with R² in brackets:

| | CH toes-down | HM toes-down | IES toes-down | CH other | HM other | IES other |
|---|---|---|---|---|---|---|
| typical peak | 18.8 | 3.2 | 6.4 | 51.9 | 13.9 | 17.4 |
| **N0** | 0.66 (0.98) | 0.53 (0.81) | 1.12 (0.84) | 1.73 (0.98) | 0.62 (0.97) | 0.70 (0.98) |
| N3 | 1.26 (0.92) | 0.60 (0.73) | **0.47 (0.97)** | 1.80 (0.98) | 0.73 (0.93) | 0.70 (0.98) |

The same story as Stage 1, with new people:

- **N3 helps one person and hurts the others.** It fixes IES's toes-down positions, where every N0 seed predicted
  about 1 Nm too much push, and it makes CH and HM worse. Its group median at toes-down looks better (0.70 against
  0.86) only because of IES. Under the plan's rule it would fail again: better for 1 person of 3, and worse elsewhere
  by 0.23 Nm.
- **The LSTM again beats ridge and the session-1 average** at every group. The ridge scalings N0, N1 and N2 again
  give the same predictions.
- **Seeds agree.** The worst of the three N0 seeds gives 0.95 Nm at toes-down and 0.85 Nm elsewhere, close to the
  medians.

The toes-down R² is much higher here (0.86) than in Stage 1 (0.47). That is not because the model got better. CH's
toes-down positions are toes-up pulls with large torques (see the data notes below), and IES and HM have no
offset as large as YES's. A higher R² at a position with more movement is the expected behaviour of R².

### Stage 2 decides the model

N0 stays. It was chosen on YES, JM and EG by a rule fixed in advance, and on three new people the only alternative
with a lower toes-down median, N3, again helped one person and hurt two. **Quote the Stage 2 N0 numbers above.**

## An extra test, outside the plan: correcting the resting level with the relaxed holds

*This idea came after seeing Stage 1, so it is exploratory. It is reported because it was tested, not because it
won.*

Before every active trial, the person sat relaxed for 30 seconds at the same angle. The difference between that
relaxed torque in session 2 and in session 1 tells us how much the resting level moved. We shifted every prediction
of that position by this difference. There is no retraining: the same N0 predictions, moved up or down. It uses
session 2's torque, but only from the relaxed recording, never from the push being predicted, so it is a calibration
step. IES has no relaxed holds, and JM and HM are each missing one angle (left uncorrected).

| N0 | toes-down MAE before | after | other MAE before | after |
|---|---|---|---|---|
| Stage 1 (YES, JM, EG) | 0.85 | **0.51** | 1.14 | 1.04 |
| Stage 2 (CH, HM only) | 0.56 | 0.48 | 0.92 | 0.92 |

| N0, toes-down MAE | before | after |
|---|---|---|
| YES | 2.37 | **0.48** |
| JM | 0.43 | 0.31 |
| EG | 0.85 | 0.71 |
| CH | 0.66 | 0.75, worse |
| HM | 0.53 | 0.37 |

**The honest reading:** it is a large fix when the resting level really moved, as it did for YES by about 2.5 Nm.
When it moved by less than about 1 Nm, as for the other four people, the result is mixed: better for JM and HM, a
little better for EG, worse for CH. A likely reason is that the relaxed hold is recorded before the trial, and the
resting level can drift a little during the trial itself. Choosing a threshold for when to apply it would need more
people than we have. So it is not part of the model, but it is the most promising lead for the
toes-down offset.

## Data notes found along the way

- **CH pulls the toes up at the toes-down positions.** At CH's three most toes-down angles, both sessions are toes-up
  pulls with the tibialis anterior, not calf pushes: torque reaches about plus 20 Nm. So CH's "toes-down" numbers
  measure something different from everyone else's.
- **CH changes direction at minus 0.33 radians.** Session 1's trial at that angle is a toes-up pull, session 2's is a
  calf push. The model handled it better than expected (MAE 1.51 Nm, R² 0.96). Leaving that trial out moves the N0
  "other positions" median from 0.76 to 0.72 Nm. The number quoted above keeps it, as the plan says.
- **IES has no relaxed holds**, and JM and HM have none at one angle.
- **The five other people push with the calf at every angle**, in both sessions.

## Limitations

- **Six people, two sessions, one trial per angle per session.** Every per-person number rests on one 90-second
  trial.
- **Stage 1 chose using session 2 of YES, JM and EG.** Stage 2 is the clean check, but it is only three people.
- **Total torque is not comparable with D01 or the baseline**, which predicted active torque.
- **N3 and the relaxed-hold correction need session 2's data before they can predict.** Neither would run on a
  device without a calibration block first. N0 needs nothing from session 2.
- **Auto-gain was not run here.** So this study cannot say how close N3 gets to auto-gain in numbers.
- **Nothing was tested in real time.**

## Where everything lives

- Plan and rule: `EXPERIMENTS/normalization/PLAN.md`. Stage 1 verdict: `decision_stage1.md`.
- Scores: `runs_stage1.csv`, `cells_stage1.csv`, `runs_stage2.csv`, `cells_stage2.csv` (one row per run and
  position, and per position over seeds). The relaxed-hold test: `relaxed_hold_extra.csv`.
- Picture of the toes-down offset: `figures/toes_down_explained.png`.
- Runs: `EXPERIMENTS/results/N0_total_trainmax` ... `NR3_total_sessionp99`, logs in `EXPERIMENTS/logs/`.
- Dashboard: sidebar group "Normalization study". This report is under "How it works" for each arm.
