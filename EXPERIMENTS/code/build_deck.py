"""
Build the presentation / report deck: EXPERIMENTS/presentation/deck.html -> deck.pdf

    python EXPERIMENTS/code/presentation_metrics.py
    python EXPERIMENTS/code/rescore_round2.py
    python EXPERIMENTS/code/deck_figures.py
    python EXPERIMENTS/code/build_deck.py            # writes deck.html and prints it to PDF with Chrome

Every number in the text is read from presentation/numbers.json and the results CSVs, so
the deck can be rebuilt when more runs finish.  Display conventions: R2 below 0 shown as
0, medians over training runs, 5th-95th / 1st-99th percentiles, MAE in Nm.
"""
import os
import sys
import json
import html
import subprocess

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
PRES = os.path.join(EXP, 'presentation')
RES = os.path.join(EXP, 'results')
SUBJ = ['YES', 'EG', 'JM', 'IES', 'CH', 'HM']
N = json.load(open(os.path.join(PRES, 'numbers.json')))
DATE = '2026-10-25'


def r2(v):
    return '–' if v is None or not np.isfinite(v) else f'{max(v, 0):.2f}'


def nm(v):
    return '–' if v is None or not np.isfinite(v) else f'{v:.2f}'


def pct(a, b):
    return f'{100 * (b / a - 1):+.0f} %'


E = html.escape
slides = []


def slide(section, title, body, cls=''):
    slides.append((section, title, body, cls))


def fig(name, h=430, cap=''):
    p = os.path.join(PRES, 'fig', name)
    if not os.path.exists(p):
        return f'<div class="pending">Figure pending: the runs for it are still training ({E(name)}).</div>'
    return f'<figure><img src="fig/{name}" style="max-height:{h}px">' + (f'<figcaption>{cap}</figcaption>' if cap else '') + '</figure>'


def bullets(items, cls=''):
    return f'<ul class="{cls}">' + ''.join(f'<li>{i}</li>' for i in items) + '</ul>'


def card(title, body, cls=''):
    return f'<div class="card {cls}"><h3>{title}</h3>{body}</div>'


def table(header, rows, cls='t'):
    h = '<tr>' + ''.join(f'<th>{c}</th>' for c in header) + '</tr>'
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<table class="{cls}">{h}{b}</table>'


O, OF, OU = N['overall']['original'], N['overall']['original_fw'], N['overall']['ours']
seeds = N['seeds']
SO, SN = seeds['original'], seeds['ours']
RT = N['runtime']

# ── data-driven wording: the deck is rebuilt when more runs finish, so claims are checked here ──
SUBN = N['subjects']
yes_pos = SUBN['YES']['positions'][min(SUBN['YES']['positions'], key=float)]
YES_POOR = yes_pos['r2_new'] < 0.5
IMPROVED = [s for s in SUBJ if SUBN[s]['td_r2_new'] > SUBN[s]['td_r2_old'] + 0.02 or SUBN[s]['td_mae_new'] < 0.9 * SUBN[s]['td_mae_old']]
NOT_IMPROVED = [s for s in SUBJ if s not in IMPROVED]
SPREAD_OLD, SPREAD_NEW = O['td_w_r2'], OU['td_w_r2']          # mean 5-95 % width over runs, per toes-down position
REPEATABLE = SPREAD_NEW < 0.6 * SPREAD_OLD
NO_LOSS_OTHER = OU['ot_r2'] >= O['ot_r2'] - 0.005 and OU['ot_mae'] <= 1.02 * O['ot_mae']
WP = N['worse_positions']
Y49 = N['yes49']
LCR = N.get('lc_ratios') or {}
VAL_FIT, TEST_VAL, TEST_VAL_OLD = LCR.get('ours_val_fit', np.nan), LCR.get('ours_test_val', np.nan), LCR.get('old_test_val', np.nan)
NO_MEMO = np.isfinite(VAL_FIT) and VAL_FIT < 1.25
SHIFT_MAIN = np.isfinite(TEST_VAL) and TEST_VAL > 1.25 and NO_MEMO
GAP_SHRANK = np.isfinite(TEST_VAL_OLD) and np.isfinite(TEST_VAL) and TEST_VAL_OLD - TEST_VAL > 0.3
LC_NOTE = 'learning curves: one training run per person; last semester’s model re-run in our framework; medians of the per-person ratios'
gap_txt = (f'Within a session the model does not memorise (check error {100 * (VAL_FIT - 1):+.0f} % vs fit). On the other session the error rises: '
           f'last semester {100 * (TEST_VAL_OLD - 1):+.0f} % vs check, ours {100 * (TEST_VAL - 1):+.0f} % ({LC_NOTE}). So the session change, not overfitting, was the main problem, and our changes shrank it.'
           if NO_MEMO and GAP_SHRANK else 'Section E shows how much of the remaining error is overfitting and how much is a shift between sessions.')
spread_txt = (f'Ours is also more repeatable: at each toes-down position the 5th–95th percentile over the {SN} runs is on average {SPREAD_NEW:.2f} wide in R² '
              f'(last semester {SPREAD_OLD:.2f}) and {OU["td_w_mae"]:.2f} Nm wide in MAE (last semester {O["td_w_mae"]:.2f} Nm).'
              if REPEATABLE else f'Run-to-run spread per toes-down position (5th–95th percentile over runs): R² {SPREAD_OLD:.2f} (last semester) vs {SPREAD_NEW:.2f} (ours).')
RUNS_TXT = (f'{SO} runs each for last semester’s model and ours (learn 1 → test 2); 3 runs in the reverse direction and for each single change; '
            '2 runs in the screening (3 for the learning rate); 3 for the 0.5 s window; 1 for the learning curves')
PR = pd.read_csv(os.path.join(RES, 'presentation_runs.csv'))       # one row per run x position (for a few direct checks below)
PR['r2f'] = PR.r2.clip(lower=0)
# first 3-run check of last semester's model (seeds 0-2): mean R2 at the toes-down positions, each run x position floored at -1
FIRST_CHECK = {s: float(PR[(PR.model == 'original') & (PR.subject == s) & (PR.seed <= 2) & (PR.toes_down == 1)].r2.clip(lower=-1).mean()) for s in SUBJ}
_R2E = pd.read_csv(os.path.join(RES, 'round2_rescored.csv'))
_R2E = _R2E[_R2E.subject.isin(['YES', 'JM', 'EG'])].assign(r2f=lambda x: x.r2.clip(lower=0))


def r2e(study, exp, toes, col='r2f'):
    """earlier exploration, deck convention: median over runs at each position, then over positions"""
    x = _R2E[(_R2E.study == study) & (_R2E.experiment == exp) & (_R2E.toes_down == toes)]
    return float(np.median(x.groupby(['subject', 'direction', 'position_rad'])[col].median())) if len(x) else np.nan
GROUP_TXT = 'group values: median over runs at each position first, then the median across the positions of the group (for one person’s 2 toes-down positions: the average of the two)'

# ═════════════════════════════════════════════════════════════════════════════
# FRONT
# ═════════════════════════════════════════════════════════════════════════════
slide('', '', f'''
<div class="titlepage">
  <div class="kicker">Capstone · EMG-based ankle torque estimation · progress report · {DATE}</div>
  <h1 class="big">Predicting voluntary ankle torque from muscle signals:<br>what we changed, how we tested it, and what it did</h1>
  <p class="lead">Same small network as last semester. We changed what goes into it (the torque we train on and the scaling of the muscle signals)
  and how training weighs the 8 ankle positions. Every change was tested on its own and then together, on all 6 people, with repeated training runs.</p>
  <p class="small">How to read this deck: before every result there is a slide that explains the approach in plain words.
  All R² values below 0 are shown as 0. Spread is shown as the median and the 5th–95th percentile of the training runs at each position;
  summaries over several positions first take the median over runs at each position, then the median across positions.
  Numbers are for "learn session 1 → test session 2" unless a slide says otherwise.</p>
</div>''', 'cover')

ld = N['ladder']
slide('Summary', 'One-page summary', f'''
<div class="cols">
<div>
{card('What we did', bullets([
    'Kept last semester’s small LSTM network (1,001 adjustable numbers; reads the last <b>0.2 s</b> of 4 muscle signals + ankle angle).',
    '<b>Change 1: clean passive table</b>: passive torque only from the still part of each passive trial (removes JM’s wrong entry).',
    '<b>Change 2: EMG auto-gain</b>: each muscle rescaled by its own recent quiet and strong levels (the same push gave a different EMG size in session 2).',
    '<b>Change 3: equal weight per position</b>: training cares about the small toes-down pushes as much as the big ones.',
    f'<b>New testing</b>: both directions; {SN} training runs (random seeds) each for the two main models in learn 1 → test 2 (3 in the reverse direction).']))}
</div>
<div>
{card('What it did (all 6 people, learn session 1 → test session 2, ' + str(SN) + ' runs each)', table(
    ['', 'last semester', 'ours'],
    [['R² at the 2 toes-down positions', r2(O['td_r2']), f"<b>{r2(OU['td_r2'])}</b>"],
     ['MAE at the 2 toes-down positions', f"{nm(O['td_mae'])} Nm", f"<b>{nm(OU['td_mae'])} Nm</b>"],
     ['R² / MAE at the other 6 positions', f"{r2(O['ot_r2'])} / {nm(O['ot_mae'])} Nm", f"<b>{r2(OU['ot_r2'])} / {nm(OU['ot_mae'])} Nm</b>"],
     ['R² toes-down, IES + HM only (not used for any decision)', r2(N['untouched_IES_HM']['original']['td_r2']), f"<b>{r2(N['untouched_IES_HM']['ours']['td_r2'])}</b>"]], 't compact') +
    f'<p class="small">Median over runs at each position, then over positions. MAE is lower at {WP["n_lower"]} of {WP["n_total"]} positions.</p>')}
{card('What did not help / what is left', bullets([
    'No clear gain from dropout or other network types. A smaller learning rate and a 0.5 s window gave smaller gains than the three changes (section C).',
    'YES −0.49 rad stays poor: learning on session 1, a resting-level shift the EMG cannot see; in the reverse direction it is worse with our model.',
    ('The main remaining error is a <b>shift between sessions</b>, not overfitting (section E).' if SHIFT_MAIN else
     ('No memorising within a session; the session change was the main problem, and our changes shrank it (section E).' if NO_MEMO and GAP_SHRANK else gap_txt))]), 'warn')}
</div></div>''')

slide('Contents', 'Contents', f'''
<div class="cols3">
{card('A · Setup', bullets(['What we predict', 'The data', 'The model', 'How we train', 'How we test (new method)', 'How we score', 'Hardware']))}
{card('B · Last semester', bullets(['Results, all 6 people', 'Why it failed at toes-down']))}
{card('C · What we changed', bullets(['Clean passive table', 'EMG auto-gain', 'Equal weight per position', 'All three together, step by step', 'Things that did not help', 'Window length (0.2 / 0.4 / 0.5 s)', 'Earlier exploration (other models, ridge …)']))}
{card('D · Final results', bullets(['R² and MAE per position, all people', 'Per person: all 8 waveforms, best and worst position', 'Best vs worst training run', 'Both directions']))}
{card('E · Bias, variance, overfitting', bullets(['What the words mean', 'Learning curves', 'Where the error comes from']))}
{card('F · Extra checks + G · Close', bullets(['Torque demeaning', 'Comparison with the 500 ms analysis', '2 papers', 'Limitations, conclusions', 'Appendix: full tables, glossary, code']))}
</div>''')

# ═════════════════════════════════════════════════════════════════════════════
# A. SETUP
# ═════════════════════════════════════════════════════════════════════════════
slides.append(('A · Setup', 'A · Setup', '<p class="lead">What we predict, the data, the model, and exactly how we train, test and score it.</p>', 'divider'))

svg_torque = '''<svg viewBox="0 0 560 250" class="diag">
<line x1="40" y1="210" x2="540" y2="210" stroke="#1b1b1b"/><line x1="40" y1="20" x2="40" y2="210" stroke="#1b1b1b"/>
<text x="10" y="120" transform="rotate(-90 14,120)" font-size="14">torque (Nm)</text><text x="460" y="232" font-size="14">time (s)</text>
<line x1="40" y1="170" x2="540" y2="170" stroke="#2a78d6" stroke-dasharray="6 4" stroke-width="2"/>
<text x="400" y="190" font-size="14" fill="#2a78d6">passive (relaxed) torque</text>
<polyline fill="none" stroke="#1b1b1b" stroke-width="2" points="40,170 90,170 100,80 160,80 170,170 230,170 240,110 290,110 300,170 350,170 360,60 430,60 440,170 540,170"/>
<text x="210" y="45" font-size="14">measured torque = passive + voluntary</text>
<line x1="395" y1="168" x2="395" y2="64" stroke="#eb6834" stroke-width="2" marker-end="url(#a)" marker-start="url(#a)"/>
<text x="455" y="110" font-size="14" fill="#eb6834">voluntary</text><text x="455" y="128" font-size="14" fill="#eb6834">(what we predict)</text>
<defs><marker id="a" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" fill="#eb6834"/></marker></defs>
</svg>'''
slide('A · Setup', 'What we predict', f'''
<div class="cols">
<div>{svg_torque}
<p class="small">The dynamometer measures the total torque at the ankle. Part of it is <b>passive</b>: stretched tissues pull even when the person is relaxed,
and this part depends on the ankle position. The rest is <b>voluntary</b>: what the muscles produce when the person pushes.</p></div>
<div>{card('Definitions we use everywhere', bullets([
    '<b>Voluntary torque</b> = measured torque − passive torque. This is the target ("answer key").',
    '<b>Passive torque</b> comes from the <b>passive trials</b>: the person relaxes at each position before the pushing trial. One value per position and session (supervisor’s rule). For YES the values are the experimenter’s worksheet readings of the same passive holds.',
    'We score each session with <b>its own</b> passive trials.',
    '<b>Ankle position</b> in radians. <b>Negative = toes down</b> (plantarflexion), positive = toes up. The "toes-down positions" are the 2 most negative ones of each person.',
    'IES has no passive trials, so for IES nothing is subtracted (raw measured torque).']))}</div>
</div>''')

slide('A · Setup', 'The data', f'''
<div class="cols">
<div>{table(['', ''], [
    ['People', '6 (YES, EG, JM, IES, CH, HM)'],
    ['Sessions', '2 in the same visit, electrodes left on: session 1 ("test") and session 2 ("retest")'],
    ['Ankle positions', '8 per session, about −0.79 to +0.25 rad'],
    ['One pushing trial', 'about 90 s at one fixed position (isometric: the foot does not move), many pushes and rests'],
    ['Signals', '4 muscle signals (EMG): medial and lateral gastrocnemius, soleus, tibialis anterior; ankle angle; torque'],
    ['EMG processing', 'band-pass 30–300 Hz → absolute value → 2 Hz smoothing ("envelope") → 100 samples per second'],
    ['Per person and session', '8 trials × 8,910 samples'],
    ['Passive trials', 'one relaxed hold per position and session (none for IES)']])}</div>
<div>{card('Why the toes-down positions are hard', bullets([
    'At the toes-down positions the calf is short and the pushes are small: the voluntary torque there varies by only <b>0.4–0.8 Nm</b> (standard deviation) for some people, against up to <b>28 Nm</b> at the strongest position.',
    'So an error of 0.5 Nm is invisible at a strong position but ruins the score at a toes-down one.',
    'The largest |voluntary torque| per position ranges from about 2.5 Nm (JM, toes-down) to 66 Nm (CH, toes-up). Result slides print it as "peak" so the MAE can be judged against it. '
    'Where the relaxed level is not 0 the peak differs from the push: larger at YES −0.49 rad (rest ≈ −2 Nm), smaller at IES’s toes-down positions (rest ≈ +2 Nm, nothing subtracted).']))}</div>
</div>''')

svg_model = '''<svg viewBox="0 0 1160 170" class="diag">
<defs><marker id="b" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#555"/></marker></defs>
<g font-size="15" text-anchor="middle">
<rect x="10" y="30" width="230" height="110" rx="10" fill="#f6f5f2" stroke="#bbb"/><text x="125" y="62" font-weight="600">Input window</text><text x="125" y="86">last 20 samples = 0.2 s</text><text x="125" y="108">4 EMG envelopes + angle</text><text x="125" y="128">(20 × 5 numbers)</text>
<rect x="290" y="45" width="150" height="80" rx="10" fill="#fde9df" stroke="#eb6834"/><text x="365" y="80" font-weight="600">LSTM, 8 cells</text><text x="365" y="102">memory over the 0.2 s</text>
<rect x="470" y="45" width="120" height="80" rx="10" fill="#f6f5f2" stroke="#bbb"/><text x="530" y="80">dropout 0.3</text><text x="530" y="102">(training only)</text>
<rect x="620" y="45" width="150" height="80" rx="10" fill="#fde9df" stroke="#eb6834"/><text x="695" y="80" font-weight="600">LSTM, 8 cells</text>
<rect x="800" y="45" width="120" height="80" rx="10" fill="#f6f5f2" stroke="#bbb"/><text x="860" y="80">dropout 0.3</text>
<rect x="950" y="30" width="200" height="110" rx="10" fill="#e5eefa" stroke="#2a78d6"/><text x="1050" y="70" font-weight="600">1 output</text><text x="1050" y="94">voluntary torque (Nm)</text><text x="1050" y="116">at the window's last sample</text>
</g>
<g stroke="#555" stroke-width="2"><line x1="240" y1="85" x2="288" y2="85" marker-end="url(#b)"/><line x1="440" y1="85" x2="468" y2="85" marker-end="url(#b)"/><line x1="590" y1="85" x2="618" y2="85" marker-end="url(#b)"/><line x1="770" y1="85" x2="798" y2="85" marker-end="url(#b)"/><line x1="920" y1="85" x2="948" y2="85" marker-end="url(#b)"/></g>
</svg>'''
slide('A · Setup', 'The model (unchanged from last semester)', f'''
{svg_model}
<div class="cols">
{card('What it is', bullets([
    'An LSTM is a network with a small memory: it reads the 20 samples in order and keeps what matters. It then outputs one number, the torque right now.',
    'It slides along the trial one sample at a time, so every 10 ms gives a new prediction.',
    '<b>1,001 adjustable numbers</b> in total. One person’s training session gives about 71,000 windows, 57,000 of them used for fitting (they overlap heavily, so they are far fewer independent samples).',
    '<b>One model per person</b>, trained on one session and tested on the other one.']))}
{card('Where it comes from', bullets([
    'Same layout and training settings as Truong et al. (2023, <i>Sensors</i> 23:3331): 2 × LSTM(8), dropout 0.3, Nadam 0.003, batch 8. They used it for EMG → knee and ankle moments while moving.',
    'Those settings were not tuned for our task, which is why we tested dropout, learning rate and window length (section C).',
    'Window length = <b>0.2 s (20 samples at 100 Hz)</b> for every model in this deck unless a slide says otherwise.']))}
</div>''')

svg_split = '''<svg viewBox="0 0 1160 210" class="diag">
<g font-size="15">
<text x="10" y="40" font-weight="600">Session 1 (one of the 8 trials)</text>
<rect x="290" y="18" width="560" height="34" fill="#1f9d55" opacity="0.8"/><text x="570" y="41" fill="white" text-anchor="middle">first 80 %: FIT (the network learns from this)</text>
<rect x="850" y="18" width="140" height="34" fill="#2a78d6" opacity="0.8"/><text x="920" y="41" fill="white" text-anchor="middle">last 20 %: CHECK</text>
<text x="10" y="112" font-weight="600">Session 2 (same position)</text>
<rect x="290" y="90" width="700" height="34" fill="#eb6834" opacity="0.85"/><text x="640" y="113" fill="white" text-anchor="middle">whole trial: TEST (never used for training or early stopping)</text>
<text x="290" y="160" fill="#555">CHECK = validation part: after every pass through the data (epoch) we measure the error here.</text>
<text x="290" y="182" fill="#555">Training stops when it has not improved for 25 epochs, and we keep the best epoch ("early stopping").</text>
<text x="290" y="204" fill="#555">Reverse direction: the same with the sessions swapped (learn session 2, test session 1).</text>
</g></svg>'''
slide('A · Setup', 'How we train', f'''
{svg_split}
<div class="cols">
{table(['Setting', 'Value (same as last semester)'], [
    ['Loss', 'mean squared error (in Nm², or scaled; see change 3)'], ['Optimiser', 'Nadam, learning rate 0.003'],
    ['Batch', '8 windows per update; windows shuffled'], ['Epochs', 'at most 200'],
    ['Learning-rate drop', '× 0.7 after 10 epochs without improvement (minimum 1e-6)'],
    ['Early stopping', 'after 25 epochs without improvement on CHECK; best epoch kept'],
    ['EMG scaling (last semester)', 'each muscle ÷ its maximum in the training session']])}
{card('In practice', bullets([
    f'Training stopped by itself in every run (at most {RT["epochs_max"]} of 200 epochs). The epoch kept was between {RT["best_epoch_min"]} and {RT["best_epoch_max"]} (median {RT["best_epoch_median"]:.0f}).',
    'Nothing from the tested session is used to fit the model: training targets, passive values for training, the starting EMG scale and early stopping all come from the training session. '
    'At prediction time, auto-gain (change 2) adjusts the EMG scale from the tested session’s own <b>past</b> EMG, never its torque.',
    'Every score is recomputed from the saved predictions by our scoring scripts (appendix: code), and an independent review of the code found no leakage of test-session torque.']))}
</div>''')

slide('A · Setup', 'How we test: the new method', f'''
<div class="cols">
{card('1 · Both directions', bullets([
    '<b>Learn session 1 → test session 2</b>: the original protocol, our main result.',
    '<b>Learn session 2 → test session 1</b>: the same with the sessions swapped. If a change only works one way, it is probably luck.']))}
{card('2 · Repeated training runs (random seeds)', bullets([
    'Training has randomness: the starting values of the 1,001 numbers, the order of the windows, and the dropout pattern. A "seed" fixes that randomness.',
    f'We train the same model on the same data with different seeds: <b>{SO} runs</b> each for last semester’s model and ours (learn 1 → test 2; 3 in the reverse direction), 3 runs for each of the three changes alone, 2 for the other screened changes (3 for the learning rate).',
    '<b>The spread we show comes from these training runs.</b> The test trials are always the same 8 trials per person.']))}
</div>
<div class="cols">
{card('3 · One change at a time, rules written first', bullets([
    'Each idea changes exactly one thing compared with last semester’s model, everything else stays fixed.',
    'The pass/fail rules were written before any result (e.g. toes-down R² up by at least 0.10, no other position worse by more than 0.05). '
    '<b>No single change met all of them</b>, mainly because of the run-to-run noise at YES −0.49 rad. We kept the final changes <b>by judgment</b>: they gave the lowest toes-down error and consistent gains at EG −0.74, EG −0.60 and YES −0.38 rad.']))}
{card('4 · Screening, then checking on new people', bullets([
    'Ideas were first screened on YES and EG, the two lowest toes-down scores in our first 3-run check of last semester’s model '
    f'(mean R² with each run and position floored at −1: YES {FIRST_CHECK["YES"]:.2f}, EG {FIRST_CHECK["EG"]:.2f}, JM {FIRST_CHECK["JM"]:.2f}; recomputed with today’s answer key).',
    'The final combination was then run on all 6 people. <b>IES and HM were not used for any decision in this study</b>, so their results are the cleanest check (report 1 had looked at them with a similar pipeline).']))}
</div>''')

svg_box = '''<svg viewBox="0 0 300 230" class="diag" style="max-width:300px">
<g font-size="13"><line x1="90" y1="30" x2="90" y2="200" stroke="#1b1b1b"/><line x1="70" y1="30" x2="110" y2="30" stroke="#1b1b1b"/><line x1="70" y1="200" x2="110" y2="200" stroke="#1b1b1b"/>
<rect x="60" y="70" width="60" height="80" fill="#eb6834" opacity="0.6" stroke="#eb6834"/><line x1="60" y1="105" x2="120" y2="105" stroke="#000" stroke-width="2.5"/>
<circle cx="90" cy="40" r="3"/><circle cx="90" cy="88" r="3"/><circle cx="90" cy="120" r="3"/><circle cx="90" cy="190" r="3"/>
<text x="130" y="34">95th percentile</text><text x="130" y="74">75th percentile</text><text x="130" y="109" font-weight="600">median</text>
<text x="130" y="154">25th percentile</text><text x="130" y="204">5th percentile</text><text x="0" y="225">dots = individual training runs</text></g></svg>'''
slide('A · Setup', 'How we score', f'''
<div class="cols">
<div>{card('Per position, per training run', bullets([
    '<b>R²</b> = 1 − (squared error of the prediction) ÷ (squared spread of the measured torque around its own average). 1 = perfect; 0 = no better than predicting the trial’s average all the time.',
    '<b>R² can go below 0</b> when the prediction is worse than that average, e.g. when the whole curve is shifted up or down. <b>We show any value below 0 as 0</b> (supervisor’s request); the raw values are kept in the files.',
    '<b>MAE</b> (mean absolute error, Nm) = the average size of the error. It replaces RMSE (supervisor’s request).',
    'Next to the MAE we give the <b>peak</b>: the largest |voluntary torque| in that test trial, so 0.3 Nm can be judged against a 3 Nm or a 60 Nm push.']))}</div>
<div class="cols" style="grid-template-columns: 300px 1fr; gap:14px">{svg_box}
{card('Over training runs', bullets([
    'We report the <b>median</b>, not the mean (one bad run cannot drag it).',
    'Spread: <b>5th–95th percentile</b> of the runs (box plots, tables) and <b>1st–99th</b> (appendix for our model, dashboard), not the standard deviation.',
    f'Note: with {SN} runs the 1st/99th percentiles are almost the worst/best run.',
    'Every person has 8 positions → 8 R² values. Summaries over several positions take the median over runs at each position first, then the median across positions; we never pool all values into one R², because big positions would hide the small ones.']))}</div>
</div>''')

slide('A · Setup', 'Hardware and computing time', f'''
<div class="cols">
{table(['', ''], [
    ['Processor', 'AMD Ryzen 7 3700X, 8 cores / 16 threads'], ['Memory (RAM)', '32 GB'],
    ['Graphics card', 'NVIDIA RTX 3070, <b>not used</b>: TensorFlow ≥ 2.11 has no GPU support on native Windows'],
    ['Operating system', 'Windows 10'], ['Software', 'Python 3.11.1, TensorFlow 2.21.0 (CPU), NumPy, pandas'],
    ['How runs were executed', 'each training run on 1 CPU thread, 12–14 runs in parallel']])}
{card('Time', bullets([
    f'One training run: median <b>{RT["median_min"]:.0f} min</b> (range {RT["min_min"]:.0f}–{RT["max_min"]:.0f} min, depending on how many epochs it needed and how busy the CPU was).',
    f'This study: <b>{RT["runs"]} training runs ≈ {RT["cpu_hours"]:.0f} CPU-hours</b> (+12 learning-curve runs), about 1–2 days of wall-clock time on this PC.',
    'Why so slow: batch size 8 means about 7,000 weight updates per epoch. We kept it so results stay comparable with last semester.',
    'Why not the GPU: for a network this small, a GPU would mostly wait for data. Running many CPU jobs in parallel was faster and also works on teammates’ MacBooks.']))}
</div>''')

# ═════════════════════════════════════════════════════════════════════════════
# B. LAST SEMESTER
# ═════════════════════════════════════════════════════════════════════════════
slides.append(('B · Last semester', 'B · Last semester', '<p class="lead">Where we started: the original code, re-run with ' + str(SO) + ' training runs per person.</p>', 'divider'))
slide('B · Last semester', 'Last semester’s model on all 6 people', fig('baseline_box_r2.png', 440) + f'''
<div class="take">{bullets([
    f'The other 6 positions (less toes-down and toes-up) are good: R² {r2(O["ot_r2"])}.',
    f'The 2 toes-down positions (shaded) are weak and unstable: R² {r2(O["td_r2"])}, and runs of the same model differ a lot: at the same position the 5th–95th percentile range over the {SO} runs is on average {O["td_w_r2"]:.2f} wide (ours: {OU["td_w_r2"]:.2f}).',
    'JM at −0.08 rad shows 0 because the model was trained on a wrong passive value there (next slide). All predictions are re-scored with this semester’s answer key, so values differ from last semester’s own report.'])}</div>''')

slide('B · Last semester', 'Why it failed at the toes-down positions: 4 causes we found', f'''
<div class="cols">
{card('1 · A wrong value in the passive table (JM)', bullets([
    'The automatic table gave JM <b>+11.7 Nm</b> at −0.07 rad (neighbours −0.5 and −3.4 Nm). The old rule accepted two pieces of passive trials in which JM was not relaxed (foot still, but torque varying with an SD of 15–18 Nm).',
    'So the training target there was about 13 Nm off.']))}
{card('2 · The EMG size changed between sessions', bullets([
    'For about the same torque, JM’s calf EMG in session 2 was only about <b>0.4×</b> its session-1 size (0.36× at −0.64 rad, 0.43× at −0.36 rad); YES’s was about 1.3× at −0.38 rad (earlier analysis, report 1).',
    'A network that learned "this much EMG = this much torque" then predicts pushes that are too small or too big.']))}
</div><div class="cols">
{card('3 · Training ignored the small positions', bullets([
    'The loss adds up squared errors in Nm². Big positions have big errors, so they decide the fit.',
    'For YES, the 2 toes-down positions hold only about <b>1 %</b> of the torque variance the loss has to explain, against about 31 % for the strongest position. So plain squared error barely rewards fitting them.']))}
{card('4 · A level shift the EMG cannot see (YES −0.49 rad)', bullets([
    'At YES −0.49 rad, after subtracting each session’s passive value, the resting level still moved by about <b>0.7 Nm</b> between sessions, while the pushes there vary by only 0.4 Nm (SD).',
    'Most of last semester’s error at the toes-down positions was such an <b>offset</b> (whole curve shifted). The shape of the predictions followed the measured torque well (correlation 0.80–0.99).']), 'warn')}
</div>''')

# ═════════════════════════════════════════════════════════════════════════════
# C. CHANGES
# ═════════════════════════════════════════════════════════════════════════════
slides.append(('C · What we changed', 'C · What we changed', '<p class="lead">Three changes, each tested on its own first. For each one: the approach, then the result.</p>', 'divider'))

slide('C · Change 1', 'Change 1 — approach: a clean passive table', f'''
<div class="cols">
{card('What we did, step by step', '<ol>' + ''.join(f'<li>{x}</li>' for x in [
    'Take the passive trial at one position (person relaxed).',
    'Find the part where the foot is still and the torque is steady: angle wobble below 0.005 rad, torque spread below 1 Nm, for at least 5 s. In practice that is the start of the hold.',
    'Average the torque there. That is the passive value for this position and session.',
    'Repeat for every position. If a position has no clean hold (JM and HM near −0.08 rad), use the straight line between its two neighbours.',
    'Training target = measured − passive value of the <b>training</b> session; answer key for scoring = measured − passive value of the <b>tested</b> session.']) + '</ol>')}
{card('Why', bullets([
    'Removes JM’s wrong entries (+11.7 Nm in session 1, +10.8 Nm in session 2): the clean value there is about −1.9 Nm.',
    'For the other people the clean table is identical to the old one (IES has none), so this change only affects JM.',
    'Passive torque falls while a joint is held stretched and recovers after release (Duong et al. 2001), and it is lower after strong contractions (Kay & Blazevich 2009). Using the same still part of every hold keeps it consistent.']))}
</div>''')

x1 = N.get('X01', {})
slide('C · Change 1', 'Change 1 — result (JM, the only person it affects)', fig('single_X01_clean_passive.png', 420) + f'''
<div class="take">{bullets([
    f'At JM {x1.get("pos", -0.078):+.2f} rad (the wrong entry): MAE <b>{nm(x1.get("mae_before"))} → {nm(x1.get("mae_after"))} Nm</b>, R² {r2(x1.get("r2_before"))} → {r2(x1.get("r2_after"))} '
    '(last semester’s model re-run in our framework vs the clean table, same 3 seeds).',
    'This fixes the <b>training target</b> (a labelling error), not the network. The old table told the network that JM’s voluntary torque there was about 13 Nm lower than it was. '
    'All models are scored against the same clean answer key, so the old model’s 12 Nm error was real.',
    'At the other positions MAE changes by at most 0.17 Nm. R² moves by up to ±0.2 at JM’s small positions: changing one position’s target retrains the shared network, and 3 runs are too few to separate this from noise.'])}</div>''')

svg_agc = '''<svg viewBox="0 0 560 230" class="diag">
<line x1="30" y1="200" x2="540" y2="200" stroke="#1b1b1b"/><line x1="30" y1="20" x2="30" y2="200" stroke="#1b1b1b"/>
<polyline fill="none" stroke="#555" stroke-width="1.8" points="30,180 70,178 80,90 120,85 130,178 180,176 190,60 230,58 240,177 300,179 310,120 350,118 360,178 420,176 430,70 480,66 490,178 540,177"/>
<line x1="30" y1="176" x2="540" y2="176" stroke="#1f9d55" stroke-width="2" stroke-dasharray="6 4"/><text x="380" y="195" font-size="13" fill="#1f9d55">floor = 10th percentile, past 20 s</text>
<line x1="30" y1="64" x2="540" y2="64" stroke="#c0392b" stroke-width="2" stroke-dasharray="6 4"/><text x="300" y="54" font-size="13" fill="#c0392b">top = 95th percentile, past 45 s</text>
<text x="40" y="225" font-size="13">scaled EMG = (EMG − floor) ÷ (top − floor), limited to 0…2</text></svg>'''
slide('C · Change 2', 'Change 2 — approach: EMG auto-gain (automatic volume control per muscle)', f'''
<div class="cols">
<div>{svg_agc}
<p class="small">Like the automatic volume control of a phone call: if the signal is weaker today, it is turned up, so the network sees the same scale as in training.</p></div>
{card('What we did', bullets([
    'Every 0.5 s, for each of the 4 muscles: <b>floor</b> = 10th percentile of its EMG over the past 20 s (its quiet level), <b>top</b> = 95th percentile over the past 45 s (its strong-push level).',
    'Scaled EMG = (EMG − floor) ÷ (top − floor), kept between 0 and 2. The range is never allowed below 0.35× the training range, so pure noise is not blown up.',
    'At the start of a trial it starts from the training session’s values. It only looks <b>backwards in time</b> and <b>never at torque</b>.',
    'It switches itself off for a person whose training pushes go both ways (above +2 Nm and below −2 Nm), because then "quiet" and "strong" are ambiguous. Only CH triggers this.',
    'Why: the same push produced a different EMG size in session 2 (cause 2). Running normalisation has been used for EMG movement classification, where it helped when training and test data came from different people (Tanaka et al. 2022, paper slide). It had not been tested across sessions or for torque, which is what we test here.']))}
</div>''')

x5 = N.get('X05_autogain', {})
# largest gap between the original code (10 runs) and the framework re-run (3 runs) at the toes-down positions
_chaos = max(((s, SUBN[s]['td_r2_old'], x5[s]['ref_td']) for s in SUBJ if s in x5), key=lambda t: abs(t[1] - t[2]))
slide('C · Change 2', 'Change 2 — result: auto-gain alone vs last semester’s model', fig('single_X05_autogain.png', 400) + f'''
<div class="take">{bullets([
    'Reference = last semester’s model re-run in our framework, same data and the same 3 seeds; only the EMG scaling differs. '
    f'(The original code gives somewhat different values because training is chaotic, e.g. {_chaos[0]} {r2(_chaos[1])} with {SO} runs vs {r2(_chaos[2])} here.)',
    'Toes-down R² per person (average of the 2 positions): ' + ', '.join(f'{s} {r2(v["ref_td"])} → <b>{r2(v["new_td"])}</b>' for s, v in x5.items()) + '.',
    'Largest gain for EG and JM. YES −0.49 rad gets worse (MAE ' +
    (lambda v: f'{nm(v["ref_pos_mae"][min(v["ref_pos_mae"], key=float)])} → {nm(v["new_pos_mae"][min(v["new_pos_mae"], key=float)])} Nm; raw R² '
     f'{v["ref_pos_r2raw"][min(v["ref_pos_r2raw"], key=float)]:.2f} → {v["new_pos_r2raw"][min(v["new_pos_r2raw"], key=float)]:.2f}')(x5['YES']) +
    '); a likely, untested reason is that YES barely activates the calf there, so rescaling also amplifies noise. CH is unchanged because auto-gain switches itself off for CH.'])}</div>''')

slide('C · Change 3', 'Change 3 — approach: equal weight per position', f'''
<div class="cols">
{card('What we did', bullets([
    'For each training position compute σ = the standard deviation of the voluntary torque there (in the training data). It measures how big the pushes are at that position, not ankle movement.',
    'Before training, divide every target by its position’s σ (σ at least 0.1 Nm). After predicting, multiply back by σ, so the output is in Nm again.',
    'At a test position the σ of the nearest training position is used. The position is an input, so this needs no test data.',
    'This is the same as weighted least squares with weight 1/σ² per position.']))}
{card('Why: share of the torque variance per position, YES', table(['position', 'σ (Nm)', 'share before', 'share after'], [
    ['−0.49 (toes-down)', '0.69', '0.1 %', '12.5 %'], ['−0.38 (toes-down)', '2.70', '1.0 %', '12.5 %'], ['−0.28', '4.59', '2.8 %', '12.5 %'],
    ['−0.17', '7.57', '7.6 %', '12.5 %'], ['−0.07', '8.56', '9.8 %', '12.5 %'], ['+0.03', '13.79', '25.3 %', '12.5 %'],
    ['+0.14', '15.16', '30.6 %', '12.5 %'], ['+0.24', '13.07', '22.8 %', '12.5 %']], 't compact') +
    '<p class="small">Share = σ² ÷ sum of all σ², i.e. how much each position contributes to the squared error the loss starts from. Before, the 2 toes-down positions held about 1 %; after, 25 %. '
    'Related idea: Kendall et al. (2018) weight each task by the inverse of its learned noise variance; we use the simpler inverse of each position’s target variance, which equalises scale.</p>')}
</div>''')

x4 = N.get('X04_equal_angle_weight', {})
slide('C · Change 3', 'Change 3 — result: equal weight alone vs last semester’s model', fig('single_X04_equal_angle_weight.png', 400) + f'''
<div class="take">{bullets([
    'Reference = last semester’s model re-run in our framework, same 3 seeds; only the weighting differs.',
    'Toes-down R² per person (average of the 2 positions): ' + ', '.join(f'{s} {r2(v["ref_td"])} → <b>{r2(v["new_td"])}</b>' for s, v in x4.items()) + '.',
    'Trade-off: with equal weight alone, the MAE at the other positions ' +
    '; '.join(part for part in (
        ('rose for ' + ', '.join(f'{s} ({100 * (v["new_mae_other"] / v["ref_mae_other"] - 1):+.0f} %)' for s, v in x4.items() if v['new_mae_other'] > 1.03 * v['ref_mae_other'])),
        ('barely changed for ' + ', '.join(f'{s} ({100 * (v["new_mae_other"] / v["ref_mae_other"] - 1):+.0f} %)' for s, v in x4.items() if 0.97 * v['ref_mae_other'] <= v['new_mae_other'] <= 1.03 * v['ref_mae_other'])),
        ('fell for ' + ', '.join(s for s, v in x4.items() if v['new_mae_other'] < 0.97 * v['ref_mae_other']))) if not part.endswith('for ')) +
    '. The step-by-step slides show the combined effect.'])}</div>''')

slide('C · All three together', 'Step by step: what each change adds (toes-down positions)', fig('ladder_r2_toesdown.png', 420) + f'''
<div class="take">{bullets([
    'Like for like: every bar comes from our framework with the same 3 seeds, starting from last semester’s model re-run there. Steps are cumulative: each bar adds one change to the previous one. '
    'Each value is the average of the 2 toes-down positions (median over runs at each).',
    'The clean passive table only changes JM; for the other people bars 1 and 2 are the same model. Auto-gain switches itself off for CH, so CH’s bar 3 equals bar 2.',
    f'With all {SN} runs (original code vs ours, all 6 people): R² {r2(O["td_r2"])} → <b>{r2(OU["td_r2"])}</b>, MAE {nm(O["td_mae"])} → <b>{nm(OU["td_mae"])} Nm</b>.'])}</div>''')
slide('C · All three together', 'Step by step: MAE at the toes-down positions', fig('ladder_mae_toesdown.png', 430) + '''
<div class="take">''' + bullets(['Same steps as the previous slide, now as error in Nm (lower is better).',
                              'The steps do not simply add up: the changes interact, so each bar shows the combination so far, not the effect of that change alone (the single-change slides show that).']) + '</div>')
slide('C · All three together', 'Step by step: the other 6 positions (did we break anything?)', fig('ladder_r2_other.png', 228) + fig('ladder_mae_other.png', 228) + f'''
<div class="take">{bullets([f'Other positions, all 6 people, {SN} runs each (original code vs ours): R² {r2(O["ot_r2"])} → {r2(OU["ot_r2"])}, MAE {nm(O["ot_mae"])} → {nm(OU["ot_mae"])} Nm. '
                          f'Without JM −0.08 rad (the passive-table fix) MAE still falls {nm(N["other_without_JM08"]["original"]["ot_mae"])} → {nm(N["other_without_JM08"]["ours"]["ot_mae"])} Nm.',
                          f'But not everywhere: MAE got worse at {len(WP["worse_any"]) - sum(1 for w in WP["worse_any"] if abs(w[1] + 0.486) < 0.01)} of the 36 other positions, '
                          'most at ' + ', '.join(f'{w[0]} {w[1]:+.2f} rad ({w[2]:.2f} → {w[3]:.2f} Nm)' for w in WP['worse10']) + '. R² fell by at most 0.01 at any of them.'])}</div>''')

sc = N.get('screening', [])
desc = {'X02a_dropout_0.0': 'switch dropout off (0 instead of 0.3)', 'X02b_dropout_0.1': 'less dropout (0.1)', 'X02c_dropout_0.5': 'more dropout (0.5)',
        'X03_lr_0.001': 'smaller learning rate (0.001 instead of 0.003)', 'X06_window_0.4s': 'longer window: 0.4 s (40 samples) instead of 0.2 s',
        'T01_passive_value_input': 'give the passive value of the trial’s session as a 6th input', 'X05_autogain': 'auto-gain (kept)', 'X04_equal_angle_weight': 'equal weight (kept)'}
why = {'X02a_dropout_0.0': 'is dropout hurting a tiny network?', 'X02b_dropout_0.1': 'same', 'X02c_dropout_0.5': 'more protection against overfitting?',
       'X03_lr_0.001': 'the learning rate matters most for LSTMs (Greff et al. 2017)', 'X06_window_0.4s': 'more history might help',
       'T01_passive_value_input': 'let the network learn the level shift', 'X05_autogain': '', 'X04_equal_angle_weight': ''}
verdict = {'X05_autogain': 'kept (by judgment)', 'X04_equal_angle_weight': 'kept (by judgment)',
           'X03_lr_0.001': 'checked with 3 runs in both directions; not kept (smaller gain than auto-gain, regressions at YES)',
           'T01_passive_value_input': 'rejected: input is built from the same table as the answer key (circular)'}
slide('C · Things that did not help', 'Approach: other single changes we screened', table(['change (one at a time)', 'how', 'why we tried it', 'runs', 'verdict'], [
    [E(r['label']), desc.get(r['exp'], ''), why.get(r['exp'], ''), str(r.get('runs', '')), verdict.get(r['exp'], 'not kept')]
    for r in sc]) + '''<p class="small">Screening = YES and EG (the two lowest toes-down scores in our first check), learn session 1 → test session 2, against last semester’s model re-run in our framework with the same seeds.
None of these changes met all the pass/fail rules written in advance; the ones kept were kept by judgment.</p>''')
_sc = {r['exp']: r for r in sc}
_worse_sc = [_sc[e]['d_mae_1'] for e in ('X02a_dropout_0.0', 'X02b_dropout_0.1', 'X02c_dropout_0.5', 'X06_window_0.4s') if e in _sc]
_lr, _ag, _ew, _pi = (_sc.get(e, {}) for e in ('X03_lr_0.001', 'X05_autogain', 'X04_equal_angle_weight', 'T01_passive_value_input'))
slide('C · Things that did not help', 'Result: screening of single changes (YES + EG)', fig('screening_changes.png', 420) + '''
<div class="take">''' + bullets([
    f'Dropout up or down and a 0.4 s window made the toes-down error larger (MAE {min(_worse_sc):+.0f} to {max(_worse_sc):+.0f} %).',
    (f'A smaller learning rate lowered the toes-down MAE with 3 runs ({_lr["d_mae_1"]:+.0f} % learn 1 → 2, {_lr["d_mae_1_rev"]:+.0f} % learn 2 → 1) but made some YES positions worse. '
     f'Its gain was smaller than auto-gain’s in both directions ({_ag["d_mae_1"]:+.0f} % / {_ag["d_mae_1_rev"]:+.0f} %) and than equal weight’s in learn 1 → 2 '
     f'({_ew["d_mae_1"]:+.0f} %; about equal in learn 2 → 1: {_ew["d_mae_1_rev"]:+.0f} %), so we did not keep it.' if _lr else ''),
    (f'The passive value as an input lowered the toes-down error ({_pi["d_mae_1"]:+.0f} %) but raised the error at the other positions ({_pi["d_mae_0"]:+.0f} % learn 1 → 2, {_pi["d_mae_0_rev"]:+.0f} % learn 2 → 1), '
     'and is circular: the input is built from the same table that defines the answer key.' if _pi else ''),
    f'Auto-gain and equal weight gave the largest drop in toes-down error ({_ag.get("d_mae_1", np.nan):+.0f} % and {_ew.get("d_mae_1", np.nan):+.0f} %) and were kept. '
    'Percentages: MAE, median over runs at each position, then over YES’s and EG’s positions.']) + '</div>')

wd = N.get('window')
WIN_RUNS = N.get('window_runs', {})
if wd:
    _wa, _wpb, _wc, _sl = N['window_all'], N['window_pos_better'], N.get('window_consistency', {}), N.get('window_slowdown', {})
    _W = 'D02_D01+window_0.5s'
    _tdm = 100 * (_wa['td_mae'][_W] / _wa['td_mae']['ours'] - 1)
    _otm = 100 * (_wa['ot_mae'][_W] / _wa['ot_mae']['ours'] - 1)
    win_sum = (f'With the same seeds for all 6 people, toes-down R² {r2(_wa["td_r2"]["ours"])} → {r2(_wa["td_r2"][_W])}, MAE {nm(_wa["td_mae"]["ours"])} → {nm(_wa["td_mae"][_W])} Nm ({_tdm:+.0f} %, lower at {_wpb["1"][0]} of {_wpb["1"][1]} toes-down positions); '
               f'other positions MAE {nm(_wa["ot_mae"]["ours"])} → {nm(_wa["ot_mae"][_W])} Nm ({_otm:+.0f} %, lower at {_wpb["0"][0]} of {_wpb["0"][1]}). '
               'Runs whose other-position MAE beats every matched 0.2 s run: ' + ', '.join(f'{s} {v["below"]}/{v["n"]}' for s, v in _wc.items()) + '. '
               + ('So the 0.5 s window gives a small, fairly consistent gain (R² about unchanged), in line with the teammate’s lower average error.' if _tdm < -3 or _otm < -3 else 'The effect is small.'))
wtxt = ('pending: the 0.5 s runs are still training' if not wd else
        win_sum + ' Toes-down median R², 0.2 s → 0.5 s: ' + ', '.join(f'{s} {r2(v["td_02"])} → {r2(v["td_05"])}' for s, v in wd.items()) +
        '. Other positions MAE: ' + ', '.join(f'{s} {nm(v["mae_ot_02"])} → {nm(v["mae_ot_05"])}' for s, v in wd.items()) + ' Nm.' +
        (f' ({WIN_RUNS} runs per person.)' if WIN_RUNS else ''))
slide('C · Window length', 'Window length: 0.2 s vs 0.4 s vs 0.5 s', f'''
<div class="cols">
{card('Approach', bullets([
    'The window is how much EMG history the network sees for one prediction: 0.2 s = 20 samples (last semester and ours), 0.4 s = 40, 0.5 s = 50.',
    'We tested 0.4 s on last semester’s model (screening, YES + EG) and, after a teammate found 0.5 s lowered his average error, <b>0.5 s on our final model</b> for all 6 people (one change: the window only).',
    'Earlier exploration also had a 0.4 s version of a passive-trial pipeline with a bigger network (16 cells × 2 with per-position experts; 3 people, faster training settings).']))}
{card('Results', bullets([
    f'0.4 s on last semester’s model (YES + EG, 2 runs): toes-down MAE {_sc.get("X06_window_0.4s", {}).get("d_mae_1", np.nan):+.0f} % (screening slide).',
    f'0.4 s in the earlier exploration (bigger network): toes-down R² {r2(r2e("passive", "E37_static_full", 1))} → {r2(r2e("passive", "E38_static_full_win40", 1))}, '
    f'other-position MAE {nm(r2e("passive", "E37_static_full", 0, "mae"))} → {nm(r2e("passive", "E38_static_full_win40", 0, "mae"))} Nm '
    f'({100 * (r2e("passive", "E38_static_full_win40", 0, "mae") / r2e("passive", "E37_static_full", 0, "mae") - 1):+.0f} %).',
    '0.5 s on our final model: a small gain at both the toes-down and the other positions, R² about unchanged. Details on the next slide.']))}
</div>''')
_yp = N.get('window_yes_pos', {})
slide('C · Window length', 'Result: 0.5 s window on our final model (same seeds)', fig('window_0.5s.png', 360) + f'''
<div class="take">{bullets([wtxt if not wd else win_sum,
    'Toes-down R² (average of the 2 positions), 0.2 s → 0.5 s: ' + ', '.join(f'{s} {r2(v["td_02"])} → {r2(v["td_05"])}' for s, v in (wd or {}).items() if s != 'YES') +
    (f'; YES: −0.49 rad below 0 with both windows, −0.38 rad {r2(_yp["02"].get("-0.381"))} → {r2(_yp["05"].get("-0.381"))}.' if _yp else '.'),
    f'Same seeds per person ({WIN_RUNS} runs); only the window differs. Cost: about {N.get("window_slowdown", {}).get("per_run", np.nan):.1f}× slower per run '
    f'({N.get("window_slowdown", {}).get("per_epoch", np.nan):.1f}× per epoch).'])}</div>''')

R2D = pd.read_csv(os.path.join(RES, 'round2_rescored.csv'))
R2D['r2f'] = R2D.r2.clip(lower=0)
R2D = R2D[R2D.subject.isin(['YES', 'JM', 'EG'])]


def r2stat(study, exp, toes, col='r2f'):
    """median over positions of the per-position median over runs (same convention as the rest of the deck)"""
    x = R2D[(R2D.study == study) & (R2D.experiment == exp) & (R2D.toes_down == toes)]
    return float(np.median(x.groupby(['subject', 'direction', 'position_rad'])[col].median())) if len(x) else np.nan


pairs = [('Ridge regression (no network) instead of the LSTM', 'weighted sum of recent EMG values and their squares, one set of weights per position', ('screen', 'E27_ridge'), ('screen', 'E02_minimal')),
         ('Ridge with auto-gain', 'same, with auto-gain EMG', ('screen', 'E28_ridge_agc'), ('screen', 'E19_minimal_agc')),
         ('Bigger LSTM with per-position corrections ("experts")', '16 cells × 2 + one small correction per training position (3,665 numbers)', ('final_check', 'FINAL'), ('ablation', 'E35_final_lstm8')),
         ('Pre-train on the other 5 people first', 'fit on everyone else, then continue on this person', ('screen', 'E20_xsub_agc'), ('screen', 'E19_minimal_agc')),
         ('EMG ÷ maximum push (MVC), whole leg', 'standard EMG normalisation instead of auto-gain', ('screen', 'E10_final_mvcglobal'), ('screen', 'E03_final')),
         ('EMG ÷ maximum push at that position', 'same, per position', ('screen', 'E11_final_mvcangle'), ('screen', 'E03_final')),
         ('Random EMG-size changes during training', 'multiply each window’s EMG, channel by channel, by a random factor (log-normal, typically ×0.6 to ×1.6)', ('screen', 'E16_final_ga50'), ('screen', 'E03_final')),
         ('× 3 extra weight on the 2 toes-down positions', 'on top of equal weight', ('screen', 'E13_final_pf3'), ('screen', 'E03_final')),
         ('× 10 extra weight', 'same, × 10', ('screen', 'E14_final_pf10'), ('screen', 'E03_final'))]
rows = []
for name, how, a, b in pairs:
    rows.append([name, how, f'{r2(r2stat(*b, 1))} → <b>{r2(r2stat(*a, 1))}</b>', f'{nm(r2stat(*b, 1, "mae"))} → <b>{nm(r2stat(*a, 1, "mae"))}</b>',
                 f'{nm(r2stat(*b, 0, "mae"))} → <b>{nm(r2stat(*a, 0, "mae"))}</b>'])
slide('C · Earlier exploration', 'Earlier exploration (last month): other models and ideas', table(
    ['idea (compared with its matching reference)', 'how', 'toes-down R²', 'toes-down MAE (Nm)', 'other MAE (Nm)'], rows, 't compact') + '''
<p class="small">Re-scored for this deck with the same answer key and conventions (median over runs at each position, then over positions; R² below 0 shown as 0). Caveats: 3 people (YES, JM, EG), both directions,
3 runs (1 for the bigger-LSTM row; ridge has no randomness, so its 3 runs are identical), faster training settings (batch 64), and earlier answer keys for training. Read the arrows as "reference → idea".
Ridge was worse at most positions; its average error is also hurt by a few absurd predictions (over 300 Nm) on EMG outside the training range, which the LSTM’s bounded output avoids.</p>''')

F = pd.read_csv(os.path.join(RES, 'families_rescored.csv'))
F['r2f'] = F.r2.clip(lower=0)
fam_names = {'H_pgain_posw': 'LSTM, 8 cells × 2 (report-1 pipeline)', 'F_gru8': 'GRU, 8 cells × 2', 'F_lstm32': 'LSTM, 32 cells × 2', 'F_mlp': 'plain network (MLP)', 'F_tcn': 'convolution network (TCN)'}
frows = []
_fm = lambda x, col: float(np.median(x.groupby(['subject', 'position_rad'])[col].median())) if len(x) else np.nan
for m in ['H_pgain_posw', 'F_gru8', 'F_lstm32', 'F_mlp', 'F_tcn']:
    x = F[F.model == m]
    frows.append([fam_names[m], r2(_fm(x[x.toes_down == 1], 'r2f')), nm(_fm(x[x.toes_down == 1], 'mae')), r2(_fm(x[x.toes_down == 0], 'r2f')), nm(_fm(x[x.toes_down == 0], 'mae'))])
slide('C · Earlier exploration', 'Earlier exploration: other network types (report 1)', f'''
<div class="cols">
<div>{table(['network type', 'toes-down R²', 'toes-down MAE', 'other R²', 'other MAE'], frows)}
<p class="small">Same data, same targets, same (faster) training settings and 3 seeds for all five; YES, JM, EG; learn session 1 → test session 2. Scored on each run’s own training target
(the older code did not store more), so compare the rows with each other, not with other slides.</p></div>
{card('What this tells us', bullets([
    'No network type was clearly better overall. GRU-8 was within the run-to-run spread of LSTM-8 (slightly better at the toes-down positions).',
    'LSTM-32 and the plain network were worse at the toes-down positions by more than the run-to-run spread; the convolution network was also worse, with some overlap. LSTM-32 was better at the other positions.',
    'The other types used the LSTM’s settings without tuning. Fair to say "not better under the same settings", not "the LSTM is the best possible".',
    'These network types were not re-tested on our final pipeline.']))}
</div>''')

# ═════════════════════════════════════════════════════════════════════════════
# D. FINAL RESULTS
# ═════════════════════════════════════════════════════════════════════════════
slides.append(('D · Final results', 'D · Final results', f'<p class="lead">Ours (clean passive table + auto-gain + equal weight) vs last semester, all 6 people, {SN} vs {SO} training runs.</p>', 'divider'))
slide('D · Final results', 'R² per position: last semester vs ours', fig('final_box_r2.png', 465) + f'''
<div class="take">{bullets([f'Toes-down (shaded): R² {r2(O["td_r2"])} → <b>{r2(OU["td_r2"])}</b> (median over runs at each position, then over the 12 toes-down positions). '
                          'For ours, the only toes-down position still at 0 is YES −0.49 rad (all runs below 0).',
                          spread_txt])}</div>''')
slide('D · Final results', 'MAE per position, with the peak torque of each position', fig('final_box_mae.png', 440) + f'''
<div class="take">{bullets([f'Toes-down MAE: {nm(O["td_mae"])} → <b>{nm(OU["td_mae"])} Nm</b>; other positions {nm(O["ot_mae"])} → <b>{nm(OU["ot_mae"])} Nm</b>. '
                          f'MAE is lower at {WP["n_lower"]} of {WP["n_total"]} positions; it is higher by more than 10 % at ' +
                          ', '.join(f'{w[0]} {w[1]:+.2f} ({w[2]:.2f} → {w[3]:.2f} Nm)' for w in WP['worse10']) + '.',
                          'Grey numbers = largest |answer-key torque| in the tested trial (Nm). Where the relaxed level is not 0 it differs from the push: larger at YES −0.49 rad (rest ≈ −2 Nm), '
                          'smaller at IES’s toes-down positions (rest ≈ +2 Nm, nothing subtracted). JM −0.08 rad (axis clipped): last semester’s 12 Nm error comes from the wrong passive entry it was trained on.'])}</div>''')

srows = []
for s in SUBJ:
    v = N['subjects'][s]
    P = v['positions']
    tp = v['td_positions']
    cells = []
    for col_old, col_new, fmt in (('r2_old', 'r2_new', r2), ('mae_old', 'mae_new', nm)):
        cells.append(' / '.join(fmt(P[p][col_old]) for p in tp) + ' → <b>' + ' / '.join(fmt(P[p][col_new]) for p in tp) + '</b>')
    srows.append([f'<b>{s}</b>' + (' *' if s in ('IES', 'HM') else ''), cells[0], cells[1],
                  f'{r2(v["ot_r2_old"])} → <b>{r2(v["ot_r2_new"])}</b>', f'{nm(v["ot_mae_old"])} → <b>{nm(v["ot_mae_new"])}</b>',
                  f'{v["best"]:+.2f} / {v["worst"]:+.2f}'])
slide('D · Final results', 'Per person: last semester → ours', table(['person', 'toes-down R² (most / 2nd)', 'toes-down MAE, Nm (most / 2nd)', 'other R²', 'other MAE (Nm)', 'our best / worst position by median R² (rad)'], srows, 't compact') + f'''
<p class="small">Median over the {SN} runs at each position (other positions: then the median of the 6); learn session 1 → test session 2. * = not used for any decision in this study (report 1 had looked at them with a similar pipeline).
What "ours" changes per person: JM all three changes; YES, EG, IES, HM auto-gain + equal weight (passive table unchanged; IES has none); CH equal weight only (auto-gain switches itself off).
Appendix: every position with 5th–95th percentiles (1st–99th also for our R²).</p>
<div class="take">{bullets([
    'Both toes-down positions improve for EG, JM, IES and HM; CH improves in MAE (its R² was already 0.97). '
    f'YES improves only at −0.38 rad; at −0.49 rad R² stays below 0 for both models (raw median {Y49["S1toS2"]["original"]["r2"]:.2f} → {Y49["S1toS2"]["ours"]["r2"]:.2f}, MAE {nm(Y49["S1toS2"]["original"]["mae"])} → {nm(Y49["S1toS2"]["ours"]["mae"])} Nm): cause 4, a level shift the EMG cannot see.'])}</div>''')

for s in SUBJ:
    v = N['subjects'][s]
    slide(f'D · {s}', f'{s}: measured vs predicted torque at all 8 positions', fig(f'wave_all_{s}.png', 520) + f'''
<p class="small">Black = measured voluntary torque (answer key). Orange = ours, blue = last semester; for each, one typical training run (the 5th of {SN} when sorted by R²) at that position.
Panel titles give the median R² and MAE over all {SN} runs, last semester → ours. Learn session 1 → test session 2.{' IES has no passive trials: measured torque is used directly.' if s == 'IES' else ''}</p>''')
    bp, wp = v['positions'][f'{v["best"]:+.3f}'], v['positions'][f'{v["worst"]:+.3f}']
    worst_note = ''
    if wp['r2_new'] < 0:
        worst_note = (f' Not improved: R² below 0 for both models. The torque there varies by only {wp["key_std"]:.1f} Nm (SD) around a relaxed level that is not 0, '
                      f'so a {nm(wp["mae_new"])} Nm error is larger than the signal’s own variation (the peak is mostly that offset).')
    elif wp['mae_new'] > 1.05 * wp['mae_old']:
        worst_note = f' Here ours is not better than last semester (MAE {nm(wp["mae_old"])} Nm).'
    elif abs(wp['mae_new'] - wp['mae_old']) <= 0.05 * wp['mae_old']:
        worst_note = ' About the same as last semester here.'
    best_note = f' (last semester {nm(bp["mae_old"])} Nm)' if bp['mae_new'] > 1.05 * bp['mae_old'] else ''
    slide(f'D · {s}', f'{s}: best and worst position (ours)', fig(f'wave_bestworst_{s}.png', 480) + f'''
<div class="take">{bullets([f'Best (highest median R²): {v["best"]:+.2f} rad, R² {r2(bp["r2_new"])}, MAE {nm(bp["mae_new"])} Nm{best_note}, peak {bp["peak"]:.1f} Nm.',
                            f'Worst (lowest median R²): {v["worst"]:+.2f} rad, R² {r2(wp["r2_new"])}, MAE {nm(wp["mae_new"])} Nm, peak {wp["peak"]:.1f} Nm.{worst_note}'] +
                           (['Positions differ by less than 0.03 in R² for this person, so "best" and "worst" are close.'] if max(P2['r2_new'] for P2 in v['positions'].values()) - min(P2['r2_new'] for P2 in v['positions'].values()) < 0.03 else
                            [f'Near-tie: {p2} rad has R² {r2(v["positions"][p2]["r2_new"])}, within 0.01 of the worst position.' for p2 in
                             [k for k in v['positions'] if k != f'{v["worst"]:+.3f}' and 0 <= v['positions'][k]['r2_new'] - wp['r2_new'] < 0.01][:1]]))}</div>''')

slide('D · Run-to-run spread', 'Best vs worst training run at the same position — ours', fig('seed_bestworst_ours.png', 520) + '''
<p class="small">Most toes-down position of each person. Green = the run with the highest R², red = the run with the lowest R², same test trial; the runs differ only in their random seed.
When the green and red lines lie on top of each other, the result does not depend on the seed. This is about repeatability, not accuracy: at YES −0.49 both runs agree, but both miss the torque level by about 0.4–0.5 Nm.</p>''')
slide('D · Run-to-run spread', 'Best vs worst training run at the same position — last semester', fig('seed_bestworst_original.png', 520) + '''
<p class="small">Same as the previous slide for last semester's model. Compare the gap between the green and red lines with the previous slide: the bigger the gap, the more a single run depends on luck.</p>''')

DN = N['directions']
DPS = N['directions_per_subject']['S2toS1']
DU = N['directions_untouched']
RP = N['rev_pos']
REV_WORSE = [s for s, v in DPS.items() if v['ours']['mae_td'] > 1.05 * v['original_fw']['mae_td'] or v['ours']['td'] < v['original_fw']['td'] - 0.05]
REV_NOGAIN = [s for s, v in DPS.items() if s not in REV_WORSE and v['ours']['mae_td'] >= 0.97 * v['original_fw']['mae_td'] and v['ours']['td'] <= v['original_fw']['td'] + 0.02]


def _rev_txt(s):
    """per-position wording for one person in learn 2 -> test 1 (re-run vs ours, same 3 seeds)"""
    a, b = RP[s]['original_fw'], RP[s]['ours']
    parts = []
    for p in sorted(a['mae'], key=float):
        ra, rb, ma, mb = a['r2f'][p], b['r2f'][p], a['mae'][p], b['mae'][p]
        verb = 'gets worse' if (mb > 1.05 * ma or rb < ra - 0.05) else ('improves' if (mb < 0.95 * ma or rb > ra + 0.02) else 'about the same')
        parts.append(f'{float(p):+.2f} rad {verb} (R² {r2(ra)} → {r2(rb)}, MAE {nm(ma)} → {nm(mb)} Nm)')
    return f'{s}: ' + '; '.join(parts)
slide('D · Both directions', 'Both directions: learn session 1 → 2 and learn session 2 → 1', fig('directions.png', 400) + f'''
<div class="take">{bullets([
    'In this figure "last semester" is the model re-run in our framework, 3 runs, in both directions (the original code only did learn 1 → test 2), and ours uses the same 3 seeds. '
    'So the learn 1 → 2 bars can differ from the per-person table (10 runs of the original code), by up to about 0.1.',
    f'Over all 6 people the toes-down gain also holds in learn 2 → test 1: R² {r2(DN["S2toS1"]["original_fw"]["td"])} → <b>{r2(DN["S2toS1"]["ours"]["td"])}</b>, '
    f'MAE {nm(DN["S2toS1"]["original_fw"]["mae_td"])} → <b>{nm(DN["S2toS1"]["ours"]["mae_td"])} Nm</b>; IES + HM together {r2(DU["S2toS1"]["original_fw"]["td"])} → {r2(DU["S2toS1"]["ours"]["td"])}.',
    ('Not for every person in learn 2 → test 1. ' + ' '.join(_rev_txt(s) + '.' for s in REV_WORSE + REV_NOGAIN)) if (REV_WORSE or REV_NOGAIN) else 'The gain holds for every person in both directions.'])}</div>''')

# ═════════════════════════════════════════════════════════════════════════════
# E. BIAS / VARIANCE / OVERFITTING
# ═════════════════════════════════════════════════════════════════════════════
slides.append(('E · Bias, variance, overfitting', 'E · Bias, variance, overfitting', '<p class="lead">What the words mean, how we checked them, and what we found.</p>', 'divider'))
slide('E · Concepts', 'Bias, variance and overfitting in plain words', f'''
<div class="cols3">
{card('Bias', '<p>The error that remains even with plenty of training: the model is too simple, or the inputs do not contain the information.</p>'
      '<p class="small"><b>How we check:</b> the error on the data the model was fitted on. If even that is high, it is bias.</p>')}
{card('Variance', '<p>How much the result changes when training changes: another random seed, or other training data.</p>'
      '<p class="small"><b>How we check:</b> the spread over repeated training runs (10 seeds), and the gap between the fitting part and the unseen check part of the same session.</p>')}
{card('Overfitting', '<p>The model memorises the training samples, noise included, instead of the general rule. Training error keeps falling while the error on unseen data rises.</p>'
      '<p class="small"><b>How we check:</b> learning curves (error per epoch on fit / check / test), early stopping, dropout and model-size tests.</p>')}
</div>
{card('One more thing that is none of these: a change between sessions', '<p>Session 2 is recorded later in the same visit (electrodes left on), after many pushes. For the same push the EMG size changed (JM about 0.4×), '
      'and the relaxed torque moved (YES −0.49 rad: about 0.7 Nm). Possible causes are skin–electrode contact, fatigue or muscle history. A model can be perfect on session 1 and still be off on session 2. '
      'This is called <b>dataset shift</b>. Regularising the network cannot fix it; auto-gain reduced it but did not remove it. (The clean passive table is a separate fix of a wrong label.)</p>', 'warn')}''')

LCr = N.get('learning_curves')
if LCr:
    L = pd.DataFrame(LCr)
    lrows = [[r.subject, r.model, f'{r.epochs} ({r.best_epoch})', nm(r.mae_fit), nm(r.mae_val), nm(r.mae_test)] for r in L.sort_values(['subject', 'model']).itertuples()]
    lc_tab = table(['person', 'model', 'epochs (kept)', 'MAE fit', 'MAE check', 'MAE test (session 2)'], lrows, 't compact')
    Lo = L[L.model == 'ours']
    lc_txt = ['One training run (seed 0) per person and model; "last semester" = the model re-run in our framework. Percentages = median over the 6 people of each person’s ratio.',
              f'Ours: MAE on the fitting part {nm(Lo.mae_fit.median())} Nm vs the unseen check part {nm(Lo.mae_val.median())} Nm ({100 * (VAL_FIT - 1):+.0f} %)' +
              (' → close, so <b>no memorising within a session</b>.' if NO_MEMO else ' → the check error is clearly higher: some overfitting within the session.'),
              f'On session 2: ours {100 * (TEST_VAL - 1):+.0f} % above the check part, last semester {100 * (TEST_VAL_OLD - 1):+.0f} %. ' +
              ('So the session change is where the extra error comes from, and our changes shrank that gap. Part of last semester’s gap is the wrong passive label at JM −0.08 rad '
               '(fit and check are scored on each model’s own training target, the session-2 error on the clean answer key).' if GAP_SHRANK else '')]
else:
    lc_tab, lc_txt = '<div class="pending">Learning-curve runs are still training.</div>', []
slide('E · Learning curves', 'Learning curves: fit vs check vs test, after every epoch', fig('learning_curves.png', 430) + f'''
<div class="take">{bullets(lc_txt)}</div>''')
slide('E · Learning curves', 'Learning curves: the numbers (epoch kept by early stopping)', f'''
<div class="cols" style="grid-template-columns: 1.3fr 1fr">
<div>{lc_tab}</div>
{card('How to read this', bullets([
    '<b>MAE fit</b>: error on the first 80 % of each training-session trial, which the network learned from.',
    '<b>MAE check</b>: error on the last 20 % of the same trials, never learned from. It decides when training stops.',
    '<b>MAE test</b>: error on the other session. It is only watched, never used to stop or choose anything.',
    'fit ≈ check → the model is not memorising its training samples (low variance). check ≪ test → the change between sessions causes the extra error (dataset shift).',
    'Errors are measured with dropout switched off, so the three columns are comparable. One training run (seed 0) per person and model.']))}
</div>''')

bv, en = N['biasvar'], N['ensemble']
slide('E · Where the error comes from', 'Where the error comes from: offset, other systematic error, run-to-run spread', fig('bias_variance.png', 400) + f'''
<div class="take">{bullets([
    f'Shares of the total squared error (Nm², summed over people and the toes-down positions, so the people with the biggest errors weigh most: '
    f'{", ".join(f"{t[0]} {100 * t[1]:.0f} %" for t in bv["original_1"]["top"])} of last semester’s, {", ".join(f"{t[0]} {100 * t[1]:.0f} %" for t in bv["ours_1"]["top"])} of ours).',
    f'Run-to-run spread (variance from the random seed) is small: ours {100 * bv["ours_1"]["var_share"]:.0f} %, last semester {100 * bv["original_1"]["var_share"]:.0f} %. '
    f'A whole-curve <b>offset</b> is {100 * bv["original_1"]["offset_share"]:.0f} % of last semester’s error but only {100 * bv["ours_1"]["offset_share"]:.0f} % of ours; the rest of our systematic error is in the shape and size of the pushes.',
    f'Averaging the {SN} runs (an "ensemble"): toes-down R² {r2(en["ours"]["td_r2_runs"])} (single runs) → {r2(en["ours"]["td_r2_ens"])}, MAE {nm(en["ours"]["td_mae_runs"])} → {nm(en["ours"]["td_mae_ens"])} Nm: barely more accurate, but the result no longer depends on the seed.'])}</div>''')

sc_d = {r['exp']: r for r in sc}
slide('E · Overfitting checks', 'Overfitting: all the checks together', table(['check', 'what we saw', 'meaning'], [
    ['Fit vs check error (same session)', (f'check {100 * (VAL_FIT - 1):+.0f} % vs fit (learning curves, ours)' if np.isfinite(VAL_FIT) else 'learning curves pending'),
     ('no memorising of the training samples' if NO_MEMO else ('some overfitting within the session' if np.isfinite(VAL_FIT) else '–'))],
    ['Early stopping', f'every run stopped by itself; kept epoch {RT["best_epoch_min"]}–{RT["best_epoch_max"]}, at most {RT["epochs_max"]} of 200 epochs',
     'stops training before the within-session check error rises; it cannot detect fitting to session-1 specifics, which only the other session shows'],
    ['Dropout 0 / 0.1 / 0.5 instead of 0.3 (last semester’s model, YES + EG, 2 runs)',
     'toes-down MAE ' + ' / '.join(f'{_sc[e]["d_mae_1"]:+.0f} %' for e in ('X02a_dropout_0.0', 'X02b_dropout_0.1', 'X02c_dropout_0.5') if e in _sc) + '; R² mixed',
     'no clear gain, so regularisation does not look like the bottleneck'],
    ['Model size', '1,001 numbers; the ≈ 57,000 fitting windows overlap heavily (a few thousand independent samples after 2 Hz smoothing); a bigger LSTM (32 cells, report 1) was worse at toes-down',
     'more capacity does not fix the toes-down positions'],
    ['Run-to-run spread', f'small share of the error for ours (≈ {100 * bv["ours_1"]["var_share"]:.0f} % at toes-down)', 'low variance from training randomness'],
    ['Check vs test (session 2) error', (f'test vs check: last semester {100 * (TEST_VAL_OLD - 1):+.0f} %, ours {100 * (TEST_VAL - 1):+.0f} % (1 run per person)' if np.isfinite(TEST_VAL) else 'learning curves pending'),
     ('the session change (dataset shift) was the main problem; our changes (auto-gain, clean passive table, equal weight) shrank it' if GAP_SHRANK else ('the main problem is the session change (dataset shift)' if SHIFT_MAIN else '–'))]]) + '''
<p class="small">Caveat: the check part is the last 20 % of the same trials, right after the fitting part. It measures fit within a session; only the other session shows how well the model transfers.</p>''')

# ═════════════════════════════════════════════════════════════════════════════
# F. EXTRA
# ═════════════════════════════════════════════════════════════════════════════
slides.append(('F · Extra checks', 'F · Extra checks', '<p class="lead">Torque demeaning, and how our results relate to the 500 ms analysis.</p>', 'divider'))
dm = N['demeaned']
slide('F · Torque demeaning', 'Torque demeaning: what it means and what it shows', f'''
<div class="cols" style="grid-template-columns: 1fr 1.25fr">
{card('Approach', bullets([
    '<b>Demeaned</b> = subtract each trial’s own average from both the measured and the predicted torque before scoring. What is left is the <b>shape</b>: when the pushes happen and how big they are relative to each other.',
    '<b>Demeaned R² high, normal R² low</b> → the shape is right but the whole curve is shifted (offset).',
    '<b>For scoring</b> this is easy and we report it here.',
    '<b>As a training target</b> it is possible, but the model then only predicts changes. To give real torque, a device needs the level from somewhere, e.g. re-zeroing during a known relaxed moment, like taring a scale. Using the test trial’s own average would use the answer, so it cannot be used for the main result.']))}
<div>{fig('demeaned.png', 300)}{bullets([
    f'Toes-down R² (all 6 people): last semester {r2(dm["original"]["td_raw"])} → {r2(dm["original"]["td_dm"])} after demeaning; ours {r2(dm["ours"]["td_raw"])} → {r2(dm["ours"]["td_dm"])}.',
    f'The bigger the jump from the coloured to the grey bar, the more of the error is a level shift. The mean of the per-person jumps is {dm["original"]["person_jump"]:+.2f} for last semester and {dm["ours"]["person_jump"]:+.2f} for ours; '
    f'for ours about {100 * dm["ours"]["jumps"]["YES"] / sum(dm["ours"]["jumps"].values()):.0f} % of it is YES ({dm["ours"]["jumps"]["YES"]:+.2f}, mainly −0.49 rad); every other person ≤ {max(v for k, v in dm["ours"]["jumps"].items() if k != "YES"):.2f}.'])}</div>
</div>''')

slide('F · The 500 ms analysis', 'How these results relate to the 500 ms analysis (20 Oct)', table(['', 'this deck', '500 ms analysis (teammate, 20 Oct)'], [
    ['Model and task', 'LSTM-8, learn session 1 → test session 2', 'same'],
    ['Window', '0.2 s (0.5 s tested as one change, next row)', '0.5 s'],
    ['Repeated runs', f"{SN} seeds for the final model, {SO} for last semester's", '10 seeds'],
    ['Spread shown as', "median and 5th–95th percentile; 1st–99th in the appendix and dashboard (supervisor's request)", 'mean ± SD'],
    ['Negative R²', 'shown as 0 (raw values kept in the files)', 'shown as is (e.g. JM −27)'],
    ['Passive torque', "clean table: JM's wrong entry removed", "his 20 Oct slides note that JM's passive value still needs review"],
    ['Targets / validation', 'each session scored with its own passive trials; IES raw torque; check part = last 20 % with no gap',
     'session-matched passive maps by default; IES from the MVC-session passive table; 2 s gap before validation'],
    ['Changes studied', 'passive table, auto-gain, weighting, dropout, learning rate, window, inputs', 'window length, passive calibration']], 't compact') + f'''
<div class="take">{bullets([
    'The two analyses are complementary: his looks at the window and run-to-run variability, ours at the inputs and the weighting.',
    'Our test of 0.5 s on our final model. ' + (win_sum if wd else 'pending'),
    "JM's very negative R² in the 500 ms results (−27, from his 20 Oct slides) is most likely the wrong passive entry (change 1), not the window.",
    'Worth adopting from his work: the test suite, the trial manifest, the passive drift audit, and the session-matched passive maps (the same principle as our answer key; values differ, e.g. for IES).'])}</div>''')

# ═════════════════════════════════════════════════════════════════════════════
# G. CLOSE
# ═════════════════════════════════════════════════════════════════════════════
slides.append(('G · Papers and conclusions', 'G · Papers and conclusions', '<p class="lead">Two papers that shaped the approach, the limitations, and what we conclude.</p>', 'divider'))
slide('G · Paper 1', 'Paper 1 — Schulte et al. (2022): trained EMG models degrade on a new day', f'''
<p class="small">Schulte RV, Zondag M, Buurke JH, Prinsen EC (2022). Multi-day EMG-based knee joint torque estimation using hybrid neuromusculoskeletal modelling and convolutional neural networks. <i>Frontiers in Robotics and AI</i> 9:869476. doi:10.3389/frobt.2022.869476</p>
<div class="cols3">
{card('What they did', bullets(['10 able-bodied people, knee and ankle movements on days 1, 2, 3 and 7.', 'EMG normalised by a maximum push measured again every session.', 'Models trained on day 1 only, tested on held-out day-1 data and on the later days.']))}
{card('What they found', bullets(['Neural network (convolution + LSTM): knee-torque error (NRMSE) <b>5.8 %</b> on day 1 → <b>10–11 %</b> on later days.', 'Hybrid CNN + musculoskeletal model: 10.0 % → 12.6–14.1 %. The pure musculoskeletal model (13.4–15.2 %) did not change between days, but was less accurate throughout.', 'Days 2, 3 and 7 did not differ from each other: the jump happens once, when the session changes.']))}
{card('Why it matters for us', bullets(['Same pattern as our check-vs-test gap: a model that looks fine within a session degrades on another session.', 'Supports testing across sessions (as we do): even with the maximum push re-measured every session, accuracy dropped on a new day. It motivates, but does not test, adaptive input scaling such as our auto-gain.', '<b>Limits:</b> knee, moving legs, a different model; different days, while our sessions are in the same visit with the electrodes left on.']))}
</div>''')
slide('G · Paper 2', 'Paper 2 — Tanaka et al. (2022): running normalisation of EMG', f'''
<p class="small">Tanaka T, Nambu I, Maruyama Y, Wada Y (2022). Sliding-window normalization to improve the performance of machine-learning models for real-time motion prediction using electromyography. <i>Sensors</i> 22(13):5005. doi:10.3390/s22135005</p>
<div class="cols3">
{card('What they did', bullets(['Each EMG sample is rescaled with the mean and spread of its own most recent window (100–500 ms), using only past data and no calibration step.', '10 people, 12 elbow-movement conditions, classifier for rest / flexion / extension.']))}
{card('What they found', bullets(['Own-person model: 77.7 % accuracy with sliding-window normalisation, 56.2 % with none, 77.2 % with a fixed scale from training (sliding and fixed were equal).', 'Model trained on 9 other people: <b>63.1 %</b> vs 41.4 % (none) and 54.4 % (fixed).', 'So it beat a fixed scale only when training and test data came from different people.']))}
{card('Why it matters for us', bullets(['The idea behind our auto-gain: rescale each muscle from its own recent past.', 'We use longer windows (20–45 s) and percentiles (quiet and strong levels), because our pushes last several seconds and a 0.5 s window would erase them.', '<b>Limits:</b> classification, not torque; elbow; no test across sessions or days, which is what we test.']))}
</div>''')

U = N['untouched_IES_HM']
slide('G · Limitations', 'Limitations (what a careful reader should know)', bullets([
    '<b>Rules fixed in advance were not met.</b> The noise and no-regression rules written before the study failed at several stages, and the final combination was chosen by judgment. '
    'The evidence for it is the result on people not used for choosing and the consistent gains, not the rules.',
    f'<b>Not a fully independent test.</b> The components were chosen by looking at session-2 results of some people (YES, EG; JM and CH influenced the auto-gain design). Only IES and HM were not used for any decision in this study (report 1 had already looked at them with a similar pipeline). On them alone: toes-down R² {r2(U["original"]["td_r2"])} → {r2(U["ours"]["td_r2"])}, MAE {nm(U["original"]["td_mae"])} → {nm(U["ours"]["td_mae"])} Nm.',
    '<b>JM −0.08 rad is a label fix:</b> it improves because the training target was corrected (change 1), not because the network is better.',
    f'<b>YES −0.49 rad is worse with our model.</b> Learning on session 1, the resting level moved 0.72 Nm against 0.40 Nm of push variation, a shift the EMG cannot show. '
    f'Learning on session 2, R² drops from {Y49["S2toS1"]["original_fw"]["r2"]:.2f} to {Y49["S2toS1"]["ours"]["r2"]:.2f} with no offset; a likely cause is that auto-gain amplifies noise in a calf that is barely active there.',
    '<b>IES</b> has no passive trials; measured torque is used as it is.',
    '<b>Offline processing:</b> the EMG smoothing filter looks about 0.25 s into the future (zero-phase), for every model, so comparisons between models stay fair. '
    'A causal (real-time) filter adds a delay of about 0.2 s, as long as the input window, so the cost could be noticeable; it has not been measured.',
    f'<b>Small study:</b> 6 people, 2 sessions in the same visit with the electrodes left on (easier than a different day). Runs: {RUNS_TXT}.']))

slide('G · Conclusions', 'Conclusions and next steps', f'''
<div class="cols">
{card('Conclusions', bullets([
    f'Changing the inputs and targets (clean passive table, auto-gain) and weighting each position equally improved the toes-down R² from {r2(O["td_r2"])} to {r2(OU["td_r2"])} and MAE from {nm(O["td_mae"])} to {nm(OU["td_mae"])} Nm. '
    f'Over the other positions there is no loss (R² {r2(O["ot_r2"])} → {r2(OU["ot_r2"])}, MAE {nm(O["ot_mae"])} → {nm(OU["ot_mae"])} Nm), although '
    + ', '.join(f'{w[0]} {w[1]:+.2f}' for w in WP['worse10']) + f' got {100 * min(w[3] / w[2] - 1 for w in WP["worse10"]):.0f}–{100 * max(w[3] / w[2] - 1 for w in WP["worse10"]):.0f} % worse in MAE.',
    'Dropout and other network types gave no clear gain; a smaller learning rate and a 0.5 s window gave smaller gains. None of these was re-tested on the final pipeline except the window.',
    'The improvement holds in both directions over all people and for IES + HM, the people not used for choosing. Not for every person: learning on session 2, '
    + ('YES −0.49 rad gets worse' if 'YES' in REV_WORSE else '') + (' and ' if 'YES' in REV_WORSE and REV_NOGAIN else '') + (f'{", ".join(REV_NOGAIN)} did not improve' if REV_NOGAIN else '') + '.',
    ('Training is now more repeatable: runs with different seeds give more similar results.' if REPEATABLE else 'Run-to-run spread is similar for both models (section D).'),
    ('The model does not overfit within a session. The remaining error is mostly a shift between sessions.' if SHIFT_MAIN else
     ('The model does not overfit within a session; the session change was the main problem, and the input fixes shrank that gap '
      f'(test vs check error: {100 * (TEST_VAL_OLD - 1):+.0f} % → {100 * (TEST_VAL - 1):+.0f} %).' if NO_MEMO and GAP_SHRANK else
      'See section E for how much of the remaining error is overfitting vs a shift between sessions.'))]))}
{card('Next steps', bullets([
    'Freeze this configuration and test it on new people or new sessions without changing anything.',
    'Use causal (real-time) filters and check how much accuracy that costs.',
    'Level shifts like YES −0.49: re-zero during a known relaxed moment, if the supervisor agrees that this is allowed.',
    'Consider the 0.5 s window: a small, fairly consistent gain for about 1.6× the training time.',
    'Try GRU-8 in the final pipeline (it was slightly better at toes-down in report 1).',
    'Report the average of several runs (ensemble) as the final prediction: barely more accurate, but the result no longer depends on the seed.']))}
</div>''')

# ═════════════════════════════════════════════════════════════════════════════
# APPENDIX
# ═════════════════════════════════════════════════════════════════════════════
slides.append(('Appendix', 'Appendix', '<p class="lead">Full per-position tables (median and 5th–95th percentile; 1st–99th also for our R²), glossary, code.</p>', 'divider'))
# per-position statistics straight from the runs, R2 floored at 0 BEFORE taking percentiles (same rule as the box plots)
A = PR[(PR.direction == 'S1toS2') & PR.model.isin(['original', 'ours'])]


def _st(x, col):
    v = x[col].values
    return dict(med=float(np.median(v)), p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)), p1=float(np.percentile(v, 1)), p99=float(np.percentile(v, 99)))


for pair in (SUBJ[0:2], SUBJ[2:4], SUBJ[4:6]):
    rows = []
    for s in pair:
        for pos in sorted(A[A.subject == s].position_rad.unique()):
            o = A[(A.model == 'original') & (A.subject == s) & ((A.position_rad - pos).abs() < 0.005)]
            n_ = A[(A.model == 'ours') & (A.subject == s) & ((A.position_rad - pos).abs() < 0.005)]
            if o.empty or n_.empty:
                continue
            ro, rn, mo, mn = _st(o, 'r2f'), _st(n_, 'r2f'), _st(o, 'mae'), _st(n_, 'mae')
            rows.append([s, f'{pos:+.2f}' + (' ⬇' if n_.toes_down.iloc[0] else ''), f'{n_.peak_torque.iloc[0]:.1f}',
                         f'{r2(ro["med"])} [{r2(ro["p5"])}–{r2(ro["p95"])}]', f'<b>{r2(rn["med"])}</b> [{r2(rn["p5"])}–{r2(rn["p95"])}] [{r2(rn["p1"])}–{r2(rn["p99"])}]',
                         f'{nm(mo["med"])} [{nm(mo["p5"])}–{nm(mo["p95"])}]', f'<b>{nm(mn["med"])}</b> [{nm(mn["p5"])}–{nm(mn["p95"])}]'])
    slide('Appendix', f'Per-position results: {pair[0]} and {pair[1]}', table(
        ['person', 'position (rad)', 'peak (Nm)', 'R² last semester [5–95 %]', 'R² ours [5–95 %] [1–99 %]', 'MAE last sem. [5–95 %]', 'MAE ours [5–95 %]'], rows, 't compact tiny') +
        '<p class="small">⬇ = toes-down position. Learn session 1 → test session 2. R² below 0 shown as 0. MAE in Nm. Peak = largest |answer-key torque| in the trial. '
        + ('YES −0.49 rad: both models below 0 (raw median R² last semester ' + f'{Y49["S1toS2"]["original"]["r2"]:.2f}, ours {Y49["S1toS2"]["ours"]["r2"]:.2f}). ' if 'YES' in pair else '')
        + ('JM −0.08 rad, last semester: the error comes from the wrong passive-table entry (change 1).' if 'JM' in pair else '') + '</p>')

slide('Appendix', 'Glossary', table(['word', 'meaning here'], [
    ['EMG (envelope)', 'electrical activity of a muscle measured on the skin; the envelope is its smoothed size over time'],
    ['Isometric', 'the joint does not move; the person pushes against a fixed footplate'],
    ['Plantarflexion / toes-down', 'foot pointed down; negative angles in our data'],
    ['Passive / voluntary torque', 'torque from stretched tissues when relaxed / torque produced by the muscles'],
    ['Epoch', 'one pass of training through all training windows'],
    ['Early stopping', 'stop when the check error has not improved for 25 epochs; keep the best epoch'],
    ['Seed', 'a number that fixes the randomness of one training run'],
    ['Dropout', 'during training only, randomly switch off a share of the network’s signals so it cannot rely on any single one'],
    ['R²', '1 − squared error ÷ squared spread of the measured torque; 1 perfect, 0 = as good as the average (below 0 shown as 0)'],
    ['MAE', 'mean absolute error in Nm'],
    ['Median / percentile', 'middle value of the runs / the value below which that % of runs fall; for groups of positions: median over runs at each position, then over the positions'],
    ['Bias / variance / overfitting', 'error that stays / spread across training runs / memorising the training data (section E)'],
    ['Dataset shift', 'the test data differ systematically from the training data (here: another session)']], 't compact'))

slide('Appendix', 'Code and reproducibility', table(['file (EXPERIMENTS/)', 'what it does'], [
    ['baseline/original_code/', 'frozen copy of last semester’s code (checksums in original_code_sha256.txt)'],
    ['code/run_original_baseline.py', 'runs the frozen original code for one person and seed'],
    ['code/run_experiment.py + configs/*.json', 'runs one change (config) for one person, direction and seed'],
    ['code/run_queue.py', 'runs many jobs in parallel, skipping finished ones'],
    ['code/score.py, code/presentation_metrics.py', 'scores every saved run (R², MAE, peak torque, percentiles)'],
    ['code/learning_curves.py', 'learning curves (fit / check / test error per epoch)'],
    ['code/rescore_round2.py', 're-scores the earlier exploration with the same conventions'],
    ['code/deck_figures.py, code/build_deck.py', 'all figures and this deck'],
    ['reports/FINAL_REPORT.md, experiment_ledger.csv', 'written report and experiment log of 9 Oct (3-run results, a different aggregation); this deck supersedes them where they differ'],
    ['SUCCESS_CRITERIA.md', 'the pass/fail rules fixed before the study'],
    ['reports/code_audit_2026-10-24.md', 'independent review of the pipeline for leakage and scoring errors (24 Oct)']], 't compact') +
    '<p class="small">Participant data (.flb files) are not part of the shared material.</p>')

# ═════════════════════════════════════════════════════════════════════════════
CSS = '''
@page { size: 1280px 720px; margin: 0 }
* { box-sizing: border-box }
html, body { margin: 0; padding: 0; background: #ffffff; -webkit-print-color-adjust: exact; print-color-adjust: exact }
body { font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; color: #1b1b1b }
.slide { width: 1280px; height: 720px; padding: 30px 50px 40px; position: relative; overflow: hidden; page-break-after: always; break-after: page; background: #fff }
.kicker { font-size: 13px; letter-spacing: .08em; text-transform: uppercase; color: #c4521f; font-weight: 600 }
h1 { font-size: 28px; margin: 2px 0 12px; font-weight: 650; line-height: 1.15 }
h1.big { font-size: 40px; margin: 18px 0 22px }
h3 { margin: 0 0 6px; font-size: 17px; color: #1b1b1b }
p, li { font-size: 17.5px; line-height: 1.38; margin: 0 0 6px }
ul, ol { margin: 0; padding-left: 20px }
li { margin-bottom: 5px }
.small, .small li { font-size: 14.5px; color: #4f4e49; line-height: 1.35 }
.lead { font-size: 21px; line-height: 1.4; max-width: 1050px }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 12px }
.cols3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 12px }
.card { background: #f6f5f2; border-radius: 10px; padding: 12px 16px; margin-bottom: 10px }
.card.warn { background: #fdf0e8; border-left: 4px solid #eb6834 }
.card li { font-size: 16.5px }
.take { background: #f6f5f2; border-radius: 10px; padding: 9px 16px 5px; margin-top: 6px }
.take li { font-size: 16px; margin-bottom: 3px }
figure { margin: 0; text-align: center }
figure img { max-width: 100%; display: block; margin: 0 auto }
figcaption { font-size: 13px; color: #4f4e49 }
.pending { border: 2px dashed #c9c7c1; border-radius: 10px; padding: 18px; color: #6b6a64; font-size: 15px; text-align: center; margin: 8px 0 }
table.t { border-collapse: collapse; width: 100%; font-size: 15.5px; margin-bottom: 8px }
table.t th { text-align: left; background: #ebe9e4; padding: 6px 8px; font-weight: 600 }
table.t td { padding: 5px 8px; border-bottom: 1px solid #e6e5e1; vertical-align: top }
table.compact td, table.compact th { padding: 4px 7px; font-size: 14.5px }
table.tiny td, table.tiny th { padding: 3px 6px; font-size: 13.6px }
.foot { position: absolute; bottom: 12px; left: 50px; right: 50px; font-size: 11.5px; color: #8a8984; display: flex; justify-content: space-between }
svg.diag { width: 100%; height: auto; display: block; margin: 0 auto 6px }
.divider { background: #1d2433; color: #fff }
.divider h1 { color: #fff; font-size: 44px; margin-top: 230px }
.divider .lead { color: #d9dce3 }
.divider .foot { color: #9aa1b0 }
.cover { background: linear-gradient(#fff, #fff) }
.titlepage { padding-top: 90px }
'''


def render():
    out = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>EMG ankle torque progress report</title><style>{CSS}</style></head><body>']
    for i, (sec, title, body, cls) in enumerate(slides, 1):
        head = '' if cls in ('cover', 'divider') else f'<div class="kicker">{E(sec)}</div><h1>{title}</h1>'
        if cls == 'divider':
            head = f'<h1>{E(title)}</h1>'
        out.append(f'<section class="slide {cls}" id="s{i}">{head}{body}<div class="foot"><span>EMG → voluntary ankle torque · capstone progress report · {DATE}</span><span>{i}</span></div></section>')
    # deck.html#s12 shows only slide 12 (used to render slide previews for checking)
    out.append('<script>if (location.hash) { document.querySelectorAll(".slide").forEach(function (s) { '
               'if ("#" + s.id !== location.hash) s.style.display = "none"; }); }</script>')
    out.append('</body></html>')
    p = os.path.join(PRES, 'deck.html')
    open(p, 'w', encoding='utf-8').write('\n'.join(out))
    return p


def to_pdf(src):
    chrome = next((c for c in (r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                               r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe') if os.path.exists(c)), None)
    if not chrome:
        print('no Chrome/Edge found; open deck.html and print to PDF')
        return
    pdf = os.path.join(PRES, 'deck.pdf')
    subprocess.run([chrome, '--headless=new', '--disable-gpu', '--no-pdf-header-footer', '--run-all-compositor-stages-before-draw',
                    '--virtual-time-budget=20000', f'--print-to-pdf={pdf}', 'file:///' + os.path.abspath(src).replace(os.sep, '/')], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    print('pdf:', pdf, os.path.getsize(pdf) if os.path.exists(pdf) else 'missing')


if __name__ == '__main__':
    src = render()
    print(len(slides), 'slides ->', src)
    if '--no-pdf' not in sys.argv:
        to_pdf(src)
