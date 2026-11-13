"""Writes notes/final_pipeline/everything_we_tried_plain.md with numbers taken from all_runs_per_position.csv."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, 'all_runs_per_position.csv'))
df['toes_down'] = 0
for (sj, ses), x in df.groupby(['subject', 'tested_session']):
    angs = sorted(x['angle_rad'].unique())[:2]
    df.loc[x.index[x['angle_rad'].isin(angs)], 'toes_down'] = 1
m = df.groupby(['key', 'study', 'config', 'subject', 'protocol', 'angle_rad', 'toes_down'], as_index=False)[['r2', 'rmse']].mean()
m['r2f'] = m['r2'].clip(lower=-1)


def score(cfg, key, study, subs=('YES', 'JM', 'EG')):
    x = m[(m.config == cfg) & (m.key == key) & (m.study == study) & m.subject.isin(subs)]
    if x.empty:
        return None
    per = x.groupby(['subject', 'protocol'])
    r2 = per['r2f'].mean().mean()
    pf = x[x.toes_down == 1].groupby(['subject', 'protocol'])['r2f'].mean().mean()
    e = x[x.toes_down == 1].groupby(['subject', 'protocol'])['rmse'].mean().mean()
    return r2, pf, e


def fmt(v):
    return ('−' if v < 0 else '') + f'{abs(v):.2f}'


def line(cfg, study='screen'):
    a, b = score(cfg, 'in_trial', study), score(cfg, 'passive_trial', study)
    if a is None:
        return '*(results pending)*'
    runs = '3 runs' if study in ('screen', 'passive') else '1 run'
    return (f'Main 3 people ({runs}). **Supervisor key** (passive trials of the tested session): average R² {fmt(b[0])}, '
            f'toes-down R² {fmt(b[1])}, toes-down error {b[2]:.2f} Nm. **In-trial key** (report 2): average R² {fmt(a[0])}, '
            f'toes-down R² {fmt(a[1])}, toes-down error {a[2]:.2f} Nm.')


S = []
w = S.append
w('# Everything we tried, in plain English')
w('')
w('Numbers come from `results_tables/all_results.xlsx` (sheet *Summary*), always for the **main 3 people (YES, JM, EG), both directions**.')
w('- **average R²**: over all 8 angles; each angle is averaged over runs, then floored at −1.')
w('- **toes-down**: the 2 most toes-down angles of each person.')
w('- **error**: RMSE in Nm.')
w('')
w('Every result is given against two answer keys:')
w('- **Supervisor key**: measured torque − passive torque from the passive trials of the session being tested. This is the one we use from now on.')
w('- **In-trial key**: measured torque − the relaxed level inside the same trial (what report 2 used).')
w('')
w('**Important:** each version scores best on the key that is built the same way as its own training answers.')
w('- Versions trained with the in-trial passive line do well on the in-trial key, but only about 0.46 at the toes-down angles on the supervisor key.')
w('- Versions trained with the passive trials (section 6) do well on the supervisor key (toes-down about 0.72), but only about 0.48 on the in-trial key.')
w('- The reason: the two keys disagree by about 0.6 Nm typically, up to 2 Nm at the toes-down angles, where the pushes are only about 1 Nm. So which version looks best at the toes-down angles depends on which key you believe, and that is exactly the question for the supervisor.')
w('')
w('---')
w('')
w('## 0. Starting point: the old pipeline (last semester)')
w('1. **Clean the EMG:** keep 30–300 Hz (where muscle signal lives), take the absolute value, and smooth with a 2 Hz filter. This gives a smooth "how active is the muscle" curve.')
w('2. **Scale the EMG:** divide each muscle by its biggest value in the training session, so it runs roughly 0 to 1.')
w('3. **Make the right answer:** voluntary torque = measured − passive. The passive value is one number per angle from the passive trials, read off an automatic table. That table had a bug for JM (+11.7 Nm at −0.07 rad).')
w('4. **Model:** a small LSTM network: 8 memory cells × 2 layers, 1,001 adjustable numbers. It reads the last 0.2 s of EMG + angle.')
w('5. **Fit:** least squares on torque in Nm.')
w('')
w(line('E01_old'))
w('')
w('## 1. Passive-torque ideas (what we subtract)')
items = [
    ('Clean passive table (static holds only)', None,
     'Use only the part of each passive trial where the foot is still and the torque is steady: angle wobble below 0.005 rad, torque spread below 1 Nm, at least 5 s. In practice that is the first ~20 s. Average it, one value per angle.',
     'Removes JM’s bad table entry. The automatic rule had accepted pieces that were not relaxed.',
     'Used inside the next versions (with a correction). The "passive trials only" versions without any correction are running now; see section 6.'),
    ('Passive correction α·P + β', 'E02_minimal',
     'Inside the training session’s pushing trials, find the relaxed moments from the EMG and measure the torque there. Fit a straight line: relaxed torque ≈ α × (passive-trial value) + β. α came out at about 0.8 for most people. Then subtract α·P + β instead of P.',
     'The relaxed torque inside a pushing trial is only about 80 % of the passive-trial value.',
     'Shown here with the old small network + equal angle weights (E02). The first-report pipeline (E03, section 3) also uses it.'),
    ('Passive from the relaxed hold right before each trial', 'E04_final_holdgain',
     'Instead of the table, use the mean of the passive-trial hold recorded just before this pushing trial (plus the correction).',
     'That hold is the closest passive measurement in time.', 'Tie with E03.'),
    ('Passive value that relaxes during the trial (drift model)', 'E06_final_drift',
     'Passive = α·P + β + γ·P·ln(1 + t/τ): starts at the hold value and slowly drops during the 90 s. τ is chosen on the training data.',
     'The relaxed level slides down during a trial (stress relaxation).', 'Tie with E03.'),
    ('Passive line through the relaxed moments of the same trial ("in-trial")', 'E22_restanchor_final',
     'From the EMG, find the moments when all 4 muscles are off. Take the median torque of each relaxed stretch and draw a smooth line through them (about 8 s smoothing). Subtract that line.',
     'Matches the relaxed level at the right moment and follows the drift. It needs no passive trials (IES has none).',
     'Our report-2 choice. The supervisor prefers passive trials, so from now on it is only a comparison.'),
    ('Removing the relaxed periods from training (or weighting them ×0.2)', 'E24_final_norest',
     'Drop, or count less, the moments when the person is relaxed.',
     'Question: are the relaxed periods just noise?',
     'Same R², but the model then predicts more fake torque while relaxed. Kept them.'),
]
for title, cfg, what, why, verdict in items:
    w(f'### {title}' + (f' (`{cfg}`)' if cfg else ''))
    w(f'- **What / how:** {what}')
    w(f'- **Why:** {why}')
    if cfg:
        w(f'- **Result:** {line(cfg)}')
    w(f'- **Verdict:** {verdict}')
    w('')
w('## 2. EMG ideas (how we scale the muscle signals)')
items = [
    ('Auto-gain (automatic volume control per muscle)', 'E19_minimal_agc',
     'Every 0.5 s, for each muscle: floor = its 10th percentile over the past 20 s (its quiet level); top = its 95th percentile over the past 45 s (its strong-push level). Output = (EMG − floor) ÷ (top − floor), kept between 0 and 2. The range is never smaller than 0.35× the training range. It starts from training values and uses only past EMG, never torque. It switches off automatically if the training trials push in both directions (some above +2 Nm, some below −2 Nm). Only CH triggers that.',
     'For the same push, the EMG size changed between sessions (e.g. JM: about half).',
     'Shown here on the old small network (E19). It is also part of E03 and the new pipeline.'),
    ('EMG ÷ maximum push (MVC), once for the whole leg', 'E10_final_mvcglobal',
     'Divide each muscle by its EMG during the maximum push, recorded once at the start of the visit.', 'Standard EMG normalisation.',
     'Worse than auto-gain: the MVC is recorded once, but the EMG size changes later.'),
    ('EMG ÷ maximum push at that angle', 'E11_final_mvcangle', 'Same, using the maximum push at the same angle.',
     'Muscles produce different EMG at different lengths.', 'Worse than auto-gain.'),
    ('Random EMG-size changes during training (augmentation)', 'E16_final_ga50',
     'During training only, multiply each muscle’s EMG in each window by a random factor (typical spread ±50 %), so the model gets used to size changes.',
     'Make the model insensitive to EMG size.', 'No gain on top of auto-gain (they fix the same problem).'),
    ('Each session scaled by its own EMG level', 'E26_final_sessp99',
     'Divide each session by its own 99th-percentile EMG.', 'Simple per-session normalisation.',
     'Needs the whole test session in advance (not allowed); breaks JM.'),
]
for title, cfg, what, why, verdict in items:
    w(f'### {title} (`{cfg}`)')
    w(f'- **What / how:** {what}')
    w(f'- **Why:** {why}')
    w(f'- **Result:** {line(cfg)}')
    w(f'- **Verdict:** {verdict}')
    w('')
w('## 3. Fit ideas (how much each angle counts)')
w('### Divide each angle by its typical swing (= weighted least squares)')
w('- **What it is:** for each angle, the **typical swing** σ is the standard deviation of the voluntary torque at that angle in the training data. It is not the ankle moving; it is how much the torque goes up and down there.')
w('  - YES: σ = 0.64 Nm at −0.49 rad (tiny pushes), 15.0 Nm at +0.14 rad (big pushes).')
w('- **How it is computed:**')
w('  - Before fitting, every target is divided by its angle’s σ (σ is at least 0.1 Nm).')
w('  - After predicting, the output is multiplied back by σ, so it is in Nm again.')
w('  - This is the same as weighted least squares with weight 1/σ² per angle.')
w('- **Why:** plain least squares in Nm lets the strong angles dominate. For YES, the 2 most toes-down angles carried only 1 % of the fit. After dividing, every angle carries 12.5 %.')
w('- **Result:** part of the first-report pipeline below (E03). When we took it out of the new pipeline (1 run), the main 3’s toes-down R² (in-trial key) fell from 0.86 to 0.71.')
w('')
w('### The first-report pipeline (`E03_final`): clean passive table × correction + auto-gain + per-angle corrections + divide by swing')
w(f'- **Result:** {line("E03_final")}')
w('')
for title, cfg, what, verdict in [
        ('×3 extra weight on the 2 most toes-down angles', 'E13_final_pf3', 'On top of the division, count the 2 most toes-down angles 3× more.', 'Tie.'),
        ('×10 extra weight', 'E14_final_pf10', 'Same with 10×.', 'Worse; the toes-down angles did not improve.')]:
    w(f'### {title} (`{cfg}`)')
    w(f'- **What / how:** {what}')
    w(f'- **Result:** {line(cfg)}')
    w(f'- **Verdict:** {verdict}')
    w('')
w('## 4. Model ideas')
items = [
    ('Per-angle corrections ("experts") + 16 memory cells', None,
     'Same LSTM type but 16 memory cells × 2 layers. The shared read-out (a weighted sum of the 16 summary numbers) gets one extra set of 16 multipliers per training angle. They start at 0 and are kept small. The ankle angle picks that angle’s set, and between two angles it blends the two. 3,665 numbers in total.',
     'The EMG→torque link changes with muscle length.', 'Small gain (about +0.01), fewer overshoot spikes. Part of E03 and the new pipeline.'),
    ('Longer window: 0.4 s instead of 0.2 s', 'E29_final_win40', 'The model looks at the last 0.4 s of EMG instead of 0.2 s.',
     'More history may help.', 'Small gain overall (mostly strong angles), none at toes-down. Next: try it with the passive-trial pipeline (running now, section 6).'),
    ('Pre-train on the other 5 people, then refit on this person', 'E20_xsub_agc',
     'First fit the model on everyone else’s data, then continue fitting on this person.', 'More data.', 'Worse: people differ too much.'),
    ('Straight-line model (ridge regression)', 'E27_ridge',
     'No network. A weighted sum of the recent EMG values (and their squares), with one set of weights per angle.',
     'Is a neural network needed at all?', 'Clearly worse. With auto-gain (E28) much better, but still behind the network.'),
    ('Old small network inside the in-trial pipeline', 'E31_restanchor_agc_lstm8',
     'In-trial passive line + auto-gain, but the old 8-cell network without per-angle corrections.', 'Simplest good version?', 'Slightly behind, with occasional spikes.'),
]
for title, cfg, what, why, verdict in items:
    w(f'### {title}' + (f' (`{cfg}`)' if cfg else ''))
    w(f'- **What / how:** {what}')
    w(f'- **Why:** {why}')
    if cfg:
        w(f'- **Result:** {line(cfg)}')
    w(f'- **Verdict:** {verdict}')
    w('')
w('## 5. Checks done in our first report (not in the tables)')
w('- **Other network types** (GRU, CNN, plain network, bigger LSTM) on the same data: none better everywhere.')
w('- **Angle-sensor leak test:** we replaced the measured angle (the footplate bends slightly when pushing) with the set angle. No change, so the model does not cheat through the angle.')
w('- **Re-zeroing each test trial with its own torque:** rejected, because it uses the answer we are trying to predict.')
w('')
w('## 6. Passive trials only (supervisor’s rule), running now')
w('Training answers use the passive trials of the training session; we score with the passive trials of the tested session. No correction from the relaxed moments.')
for title, cfg in [('Old pipeline with the clean passive table (JM bug removed)', 'E36_static_old'),
                   ('Passive trials only + auto-gain + per-angle corrections + divide by swing', 'E37_static_full'),
                   ('Same with a 0.4 s window', 'E38_static_full_win40')]:
    w(f'- **{title}** (`{cfg}`): {line(cfg, "passive")}')
w('')
w('## 7. All versions in one table (main 3 people, both directions)')
w('| version | runs | supervisor key: average R² | supervisor key: toes-down R² | supervisor key: toes-down error (Nm) | in-trial key: average R² | in-trial key: toes-down R² |')
w('|---|---|---|---|---|---|---|')
for study in ('screen', 'ablation', 'passive'):
    for cfg in sorted(df[df.study == study].config.unique()):
        a, b = score(cfg, 'in_trial', study), score(cfg, 'passive_trial', study)
        if a is None:
            continue
        w(f'| `{cfg}` | {"3" if study == "screen" else "1" if study == "ablation" else "3"} | {fmt(b[0])} | {fmt(b[1])} | {b[2]:.2f} | {fmt(a[0])} | {fmt(a[1])} |')
open(os.path.join(HERE, '..', 'everything_we_tried_plain.md'), 'w', encoding='utf-8', newline='\n').write('\n'.join(S) + '\n')
print('written')
