"""Pipeline tree for the weekly-update deck: one person's data from recording to score.

Every trace is real: EG, push trials at -0.04 rad (session 1 trial 22, session 2 trial 38),
passive holds at the same angle (trials 21 and 37), and the predictions of the seed-0 model
of the "0.5 s window + retest map" collection. Writes a self-contained HTML page sized
1664 x 700; render it to PNG with headless Edge (see README).

Usage (repository root on PYTHONPATH):
    python pipeline_tree.py <data dir> <overrides.json> <EG run folder> <output.html> [new_method_EG.npz]
With the .npz (from new_method_data.py) it draws the same tree for the new method instead.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ML.preprocessing.calibration import PassiveCalibration
from ML.preprocessing.emg_envelope import extract_envelope
from ML.preprocessing.pipeline import PipelineConfig, process_block, raw_blocks
from ML.preprocessing.trial_manifest import CHANNELS, load_inventory

data_dir, overrides, run, out = sys.argv[1:5]
run = Path(run)
SUBJECT, PUSH, HOLD = 'EG', {'test': 22, 'retest': 38}, {'test': 21, 'retest': 37}
SPAN = {'test': (30.0, 46.0), 'retest': (28.0, 44.0)}      # seconds shown from each push trial
WINDOW_END = {'test': 36.4, 'retest': 35.1}                # the window drawn as a heat map
STAGES = {'test': (35.0, 38.0), 'retest': (34.0, 37.0)}    # zoom for the filter stages

INK, MUTED, FAINT, LINE = '#1B2430', '#5B6573', '#9AA3AE', '#D9D3C7'
S1, S2, AMBER, TEAL = '#2F6DB5', '#C2410C', '#B45309', '#1F7A72'
MUSCLE = dict(zip(CHANNELS, ['#1F4E79', '#2F6DB5', '#6FA8DC', '#C2410C']))
LABEL = {'gm': 'GM', 'gl': 'GL', 'sol': 'SOL', 'ta': 'TA'}

# ----------------------------------------------------------------------------- data
cfg = PipelineConfig(window=50)
trials, manifest = load_inventory(SUBJECT, data_dir, overrides)
where = {r['trial']: (i, r) for i, r in enumerate(manifest['trials'])}
pre = json.loads((run / f'{SUBJECT}_preprocessing.json').read_text())
scales = np.asarray(pre['emg_scales'])
maps = {s: PassiveCalibration(**c) for s, c in pre['passive_by_session'].items()}
metrics = json.loads((run / f'{SUBJECT}_lstm_seed0' / 'metrics.json').read_text())
score = metrics['splits']['test']['trials'][f'{SUBJECT}/{PUSH["retest"]}']

D = {}
for session in ('test', 'retest'):
    i, record = where[PUSH[session]]
    frame = trials[i]
    _, a, b = raw_blocks(record, cfg)[0]           # session 1: the training block; session 2: all of it
    sig, index = process_block(frame, a, b, cfg)
    sig[:, :4] /= scales                           # session-1 training maximum, in both sessions
    t = index / 1000.0
    lo, hi = SPAN[session]
    keep = (t >= lo) & (t <= hi)
    passive = maps[session].predict(sig[:, 4])
    raw_t = np.arange(len(frame)) / 1000.0
    raw_keep = (raw_t >= lo) & (raw_t <= hi)
    end = int(np.argmin(abs(t - WINDOW_END[session])))
    split = 'train' if session == 'test' else 'test'
    z = np.load(run / f'{SUBJECT}_lstm_seed0' / f'{split}_predictions.npz')
    zk = (z['trial'] == PUSH[session]) & (z['sample_index'] / 1000.0 >= lo) & (z['sample_index'] / 1000.0 <= hi)
    hi_, hr = where[HOLD[session]]
    hold = next(e for e in maps[session].evidence if e['trial'] == HOLD[session])
    hold_frame = trials[hi_]
    D[session] = dict(
        t=t[keep], env=sig[keep, :4], angle=sig[keep, 4], measured=sig[keep, 5], passive=passive[keep],
        target=sig[keep, 5] - passive[keep],
        raw_t=raw_t[raw_keep], raw={c: frame[c].to_numpy()[raw_keep] - frame[c].to_numpy()[a:b].mean() for c in CHANNELS},
        raw_torque=frame.torque.to_numpy()[raw_keep],
        window=sig[end - cfg.window + 1:end + 1, :5], window_t=(t[end - cfg.window + 1], t[end]),
        pred_t=z['sample_index'][zk] / 1000.0, pred=z['y_pred'][zk], true=z['y_true'][zk],
        hold_t=np.arange(0, 20000) / 1000.0, hold_torque=hold_frame.torque.to_numpy()[:20000],
        hold=hold, angle_now=float(np.median(sig[keep, 4])),
        map_now=float(maps[session].predict([np.median(sig[keep, 4])])[0]))

STAGE = {}
for session in ('test', 'retest'):
    i, record = where[PUSH[session]]
    _, a, b = raw_blocks(record, cfg)[0]
    env, stages = extract_envelope(trials[i]['sol'].to_numpy()[a:b], return_intermediates=True)
    st = (np.arange(a, b) / 1000.0)
    sk = (st >= STAGES[session][0]) & (st <= STAGES[session][1])
    STAGE[session] = dict(t=st[sk], filtered=stages['filtered'][sk], rectified=stages['rectified'][sk],
                          envelope=env[sk])

for s in D:
    d = D[s]
    print(f"{s}: angle {d['angle_now']:+.3f} rad, map {d['map_now']:+.2f} Nm, hold trial {HOLD[s]} "
          f"{d['hold']['position']:+.3f} rad {d['hold']['torque']:+.2f} Nm, preds {len(d['pred'])}, "
          f"target check {np.nanmax(abs(np.interp(d['pred_t'], d['t'], d['target']) - d['true'])):.3f} Nm")
print(f"score trial {PUSH['retest']}: R2 {score['r2']:.3f}, MAE {score['mae']:.2f} Nm")

# New method (optional 5th argument, the .npz from new_method_data.py): auto-gained inputs,
# 0.2 s windows, its own passive table and the predictions of its seed-0 model replace the
# model path; the raw recording, filter stages and passive holds stay as above.
NEW = len(sys.argv) > 5
OLD_SCORE = score
if NEW:
    z = np.load(sys.argv[5])
    for session, s in (('test', 'S1'), ('retest', 'S2')):
        d, t = D[session], z[f'{s}_t']
        lo, hi = SPAN[session]
        keep = (t >= lo) & (t <= hi)
        end = int(np.argmin(abs(t - WINDOW_END[session])))
        feats = np.column_stack([z[f'{s}_agc'], z[f'{s}_pos']])
        steps = 20
        idx, ang, mean, dur = z[f'{s}_holds'][0]
        d.update(t=t[keep], env=z[f'{s}_agc'][keep], angle=z[f'{s}_pos'][keep], measured=z[f'{s}_tq'][keep],
                 passive=z[f'{s}_passive'][keep], target=z[f'{s}_target'][keep],
                 window=feats[end - steps + 1:end + 1], window_t=(t[end - steps + 1], t[end]),
                 hold=dict(position=float(ang), torque=float(mean), interval=[0, dur * 1000], stat='mean'),
                 angle_now=float(np.median(z[f'{s}_pos'][keep])), map_now=float(np.median(z[f'{s}_passive'][keep])))
        maps[session] = SimpleNamespace(positions=z[f'map_{s}'][:, 0], torques=z[f'map_{s}'][:, 1])
    pk = (z['S2_pred_t'] >= SPAN['retest'][0]) & (z['S2_pred_t'] <= SPAN['retest'][1])
    D['retest'].update(pred_t=z['S2_pred_t'][pk], pred=z['S2_pred'][pk], true=z['S2_pred_target'][pk])
    score = dict(r2=float(z['S2_r2']), mae=float(z['S2_mae']))
    KNOTS, SD = z['knots'], z['sd']
    print(f"new method: map {D['test']['map_now']:+.2f} / {D['retest']['map_now']:+.2f} Nm, "
          f"R2 {score['r2']:.3f}, MAE {score['mae']:.2f} Nm, swing here {SD[np.argmin(abs(KNOTS - D['test']['angle_now']))]:.2f} Nm")

# ----------------------------------------------------------------------------- svg helpers
def xy(t, y, box, xlim, ylim):
    x0, y0, w, h = box
    X = x0 + (np.asarray(t) - xlim[0]) / (xlim[1] - xlim[0]) * w
    Y = y0 + h - (np.clip(y, *ylim) - ylim[0]) / (ylim[1] - ylim[0]) * h
    return X, Y


def path(t, y, box, xlim, ylim, color, width=1.6, dash=None, opacity=1):
    X, Y = xy(t, y, box, xlim, ylim)
    d = 'M' + ' L'.join(f'{p:.1f} {q:.1f}' for p, q in zip(X, Y))
    extra = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" '
            f'stroke-linecap="round" opacity="{opacity}"{extra}/>')


def band(t, y, box, xlim, ylim, color, bins, width=1.0, opacity=1):
    """Raw-looking trace: the min and max of each pixel column."""
    edges = np.linspace(xlim[0], xlim[1], bins + 1)
    idx = np.digitize(t, edges) - 1
    segs = []
    for k in range(bins):
        v = y[idx == k]
        if len(v):
            X, Y = xy([(edges[k] + edges[k + 1]) / 2] * 2, [v.min(), v.max()], box, xlim, ylim)
            segs.append(f'M{X[0]:.1f} {Y[1]:.1f}V{max(Y[0], Y[1] + 0.6):.1f}')
    return f'<path d="{"".join(segs)}" stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>'


def text(x, y, s, size=11, color=MUTED, anchor='start', weight=500):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def hline(x0, x1, y, color=LINE, width=1, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x0:.1f}" x2="{x1:.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="{width}"{extra}/>'


def ramp(v):
    stops = [(0.0, (244, 247, 251)), (0.25, (190, 212, 238)), (0.55, (85, 140, 205)), (1.0, (19, 41, 75))]
    v = float(np.clip(v, 0, 1))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if v <= p1:
            f = (v - p0) / (p1 - p0)
            return '#%02x%02x%02x' % tuple(round(a + (b - a) * f) for a, b in zip(c0, c1))
    return '#13294b'


def svg(w, h, body):
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
            + ''.join(body) + '</svg>')


# ----------------------------------------------------------------------------- mini plots
PW = 216      # plot width inside a node


def raw_plot(s, h):
    d = D[s]
    xlim = SPAN[s]
    body, lab = [], 30
    rows = [h * f for f in (0.13, 0.13, 0.13, 0.13)]
    y = 2
    for c, rh in zip(CHANNELS, rows):
        v = d['raw'][c]
        lim = np.percentile(abs(v), 99.7)
        body.append(band(d['raw_t'], v, (lab, y, PW - lab, rh), xlim, (-lim, lim), MUSCLE[c], 180, 0.9))
        body.append(text(0, y + rh / 2 + 4, LABEL[c], 10.5, MUSCLE[c], weight=700))
        y += rh + 1.5
    th = h - y - 2
    body.append(path(d['raw_t'][::20], d['raw_torque'][::20], (lab, y + 2, PW - lab, th - 2), xlim, (-29, 1), INK, 1.4))
    body.append(text(0, y + th / 2 + 4, 'Nm', 10.5, INK, weight=700))
    return svg(PW, h, body)


def stage_plot(s, h):
    t, xlim = STAGE[s]['t'], STAGES[s]
    lab, rh = 66, (h - 6) / 3
    f, r, e = STAGE[s]['filtered'], STAGE[s]['rectified'], STAGE[s]['envelope']
    lim = np.percentile(abs(f), 99.8)
    body = [
        band(t, f, (lab, 1, PW - lab, rh - 2), xlim, (-lim, lim), MUSCLE['sol'], 150, 0.9),
        text(0, rh / 2 + 4, '30–300 Hz', 10.5, INK, weight=600),
        band(t, r, (lab, rh + 3, PW - lab, rh - 2), xlim, (0, lim), MUSCLE['sol'], 150, 0.9),
        text(0, rh * 1.5 + 6, 'rectify', 10.5, INK, weight=600),
        band(t, r, (lab, 2 * rh + 5, PW - lab, rh - 2), xlim, (0, lim), '#C9D6E6', 150, 0.9),
        path(t[::5], e[::5], (lab, 2 * rh + 5, PW - lab, rh - 2), xlim, (0, e.max() * 1.1), '#1F4E79', 2),
        text(0, rh * 2.5 + 8, 'smooth 2 Hz', 10.5, INK, weight=600),
    ]
    return svg(PW, h, body)


def env_plot(s, h, shade=True):
    d = D[s]
    xlim = SPAN[s]
    lab = 30
    rh = (h - 16) / 4
    body = []
    if shade:
        x0, _ = xy([d['window_t'][0] - 0.005], [0], (lab, 0, PW - lab, h), xlim, (0, 1))
        x1, _ = xy([d['window_t'][1] + 0.005], [0], (lab, 0, PW - lab, h), xlim, (0, 1))
        body.append(f'<rect x="{x0[0]:.1f}" y="0" width="{max(x1[0] - x0[0], 3):.1f}" height="{h}" fill="{TEAL}" opacity="0.22" rx="1.5"/>')
    y = 1
    for k, c in enumerate(CHANNELS):
        body.append(path(d['t'], d['env'][:, k], (lab, y, PW - lab, rh - 1.5), xlim, (0, max(0.35, d['env'][:, k].max() * 1.05)), MUSCLE[c], 1.5))
        body.append(text(0, y + rh / 2 + 4, LABEL[c], 10.5, MUSCLE[c], weight=700))
        y += rh
    body.append(path(d['t'], d['angle'], (lab, y + 2, PW - lab, 10), xlim, (-0.3, 0.2), MUTED, 1.4, '3 2'))
    body.append(text(0, y + 11, 'angle', 10.5, MUTED, weight=700))
    return svg(PW, h, body)


def window_plot(s, h):
    d = D[s]
    W = d['window']
    lab, cols, rows = 26, len(W), 5
    cell_w = 170 / cols
    cell_h = (h - 34) / rows
    cw, ch = cols * cell_w, rows * cell_h
    body = []
    for k, off in enumerate((12, 6)):
        body.append(f'<rect x="{lab + off}" y="{off * 0.8:.1f}" width="{cw}" height="{ch}" rx="3" '
                    f'fill="{["#E4ECF6", "#C9D9EE"][k]}" stroke="#AFC3DD" stroke-width="0.8"/>')
    y0 = 12 * 0.8 + 2
    for r in range(rows):
        for c in range(cols):
            v = W[c, r]
            color = ramp(v) if r < 4 else '#B8BEC6'
            body.append(f'<rect x="{lab + c * cell_w:.2f}" y="{y0 + r * cell_h:.2f}" width="{cell_w - 0.7:.2f}" '
                        f'height="{cell_h - 0.8:.2f}" fill="{color}"/>')
        name = LABEL[CHANNELS[r]] if r < 4 else 'angle'
        body.append(text(0, y0 + r * cell_h + cell_h / 2 + 3.5, name if r < 4 else '∠', 10, MUSCLE[CHANNELS[r]] if r < 4 else MUTED, weight=700))
    body.append(f'<rect x="{lab}" y="{y0:.1f}" width="{cw}" height="{ch - 0.8:.1f}" fill="none" stroke="{INK}" stroke-width="1" rx="1"/>')
    yb = y0 + ch + 6
    body.append(f'<path d="M{lab} {yb} v4 H{lab + cw} v-4" fill="none" stroke="{MUTED}" stroke-width="1"/>')
    body.append(text(lab + cw / 2, yb + 15, f'{cols} steps = {cols / 100:.1f} s', 10.5, INK, 'middle', 600))
    return svg(PW, h, body)


def lstm_plot(h, frozen=False, steps=50):
    stroke = '#7E8A99' if frozen else S1
    fill1, fill2 = ('#E6E9ED', '#D3D8DE') if frozen else ('#DCE7F5', '#B9CFEC')
    xs = [24, 56, 112, 144]
    yl2, yl1, yin = h * 0.10, h * 0.42, h * 0.76
    cw, ch = 24, 16
    body = ['<defs><marker id="m%s" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto">'
            '<path d="M0 0 L6 3 L0 6 Z" fill="%s"/></marker></defs>' % (int(frozen), stroke)]
    arrow = f'stroke="{stroke}" stroke-width="1.2" marker-end="url(#m{int(frozen)})"'
    for k, x in enumerate(xs):
        body.append(f'<rect x="{x}" y="{yin:.1f}" width="{cw}" height="{ch - 4}" rx="2" fill="#F4F7FB" stroke="#AFC3DD"/>')
        for r in range(5):
            body.append(f'<rect x="{x + 2 + r * 4}" y="{yin + 2:.1f}" width="3.2" height="{ch - 8}" fill="{ramp([0.2, 0.5, 0.7, 0.1, 0.3][r] if r < 4 else 0)}"/>')
        body.append(f'<line x1="{x + cw / 2}" y1="{yin:.1f}" x2="{x + cw / 2}" y2="{yl1 + ch + 1:.1f}" {arrow}/>')
        body.append(f'<rect x="{x}" y="{yl1:.1f}" width="{cw}" height="{ch}" rx="4" fill="{fill1}" stroke="{stroke}" stroke-width="1.2"/>')
        body.append(f'<line x1="{x + cw / 2}" y1="{yl1:.1f}" x2="{x + cw / 2}" y2="{yl2 + ch + 1:.1f}" {arrow}/>')
        body.append(f'<rect x="{x}" y="{yl2:.1f}" width="{cw}" height="{ch}" rx="4" fill="{fill2}" stroke="{stroke}" stroke-width="1.2"/>')
        body.append(text(x + cw / 2, h - 1, [f't−{steps - 1}', f't−{steps - 2}', 't−1', 't'][k], 9.5, MUTED, 'middle', 600))
    for x0, x1 in ((xs[0], xs[1]), (xs[2], xs[3])):
        for y in (yl1, yl2):
            body.append(f'<line x1="{x0 + cw}" y1="{y + ch / 2:.1f}" x2="{x1 - 1}" y2="{y + ch / 2:.1f}" {arrow}/>')
    for y in (yl1, yl2):
        body.append(f'<line x1="{xs[1] + cw}" y1="{y + ch / 2:.1f}" x2="{xs[2] - 1}" y2="{y + ch / 2:.1f}" stroke="{stroke}" stroke-width="1.2" stroke-dasharray="2 3"/>')
    body.append(text(xs[1] + cw + 18, yin + 10, '⋯', 12, MUTED, 'middle', 700))
    for y, name in ((yl2, 'L2'), (yl1, 'L1'), (yin - 2, 'in')):
        body.append(text(0, y + ch / 2 + 4, name, 10, stroke if name != 'in' else MUTED, weight=700))
    body.append(text(xs[3] + cw + 6, yl1 + ch / 2 + 4, '8 units', 10, stroke, weight=600))
    out_y = yl2 + ch / 2
    body.append(f'<line x1="{xs[3] + cw}" y1="{out_y:.1f}" x2="{193}" y2="{out_y:.1f}" {arrow}/>')
    body.append(f'<circle cx="{204}" cy="{out_y:.1f}" r="10" fill="{TEAL if not frozen else "#5E8F8A"}"/>')
    body.append(text(204, out_y + 4, 'ŷ', 11.5, '#FFFFFF', 'middle', 700))
    return svg(PW, h, body)


def trace_plot(s, h, which, w=PW, legend=False, zero=True):
    d = D[s]
    xlim = SPAN[s]
    ylim = (-24, 3)
    body = []
    box = (2, 3, w - 4, h - 6)
    if zero:
        _, y0 = xy([0], [0], box, xlim, ylim)
        body.append(hline(2, w - 2, y0[0], LINE, 1))
    if 'target' in which:
        body.append(path(d['t'], d['target'], box, xlim, ylim, TEAL, 2.2))
    if 'pred' in which:
        body.append(path(d['pred_t'], d['pred'], box, xlim, ylim, INK, 1.5, '4 2.5' if 'target' in which else None))
        if 'target' not in which:
            X, Y = xy(d['pred_t'][-1:], d['pred'][-1:], box, xlim, ylim)
            body.append(f'<circle cx="{X[0]:.1f}" cy="{Y[0]:.1f}" r="3.5" fill="{INK}"/>')
    return svg(w, h, body)


def measured_plot(s, h):
    d = D[s]
    xlim = SPAN[s]
    ylim = (-29, 2)
    box = (2, 3, PW - 4, h - 6)
    color = S1 if s == 'test' else S2
    _, yp = xy([0], [d['map_now']], box, xlim, ylim)
    return svg(PW, h, [
        path(d['t'], d['measured'], box, xlim, ylim, MUTED, 1.6),
        path(d['t'], d['passive'], box, xlim, ylim, color, 2.4, '5 3'),
        text(PW - 4, yp[0] - 5, f'map {d["map_now"]:+.1f} Nm'.replace('-', '−'), 10.5, color, 'end', 700),
    ])


def hold_plot(s, h):
    d = D[s]
    hold = d['hold']
    med = hold['torque']
    xlim, ylim = (0, 20), (med - 1.3, med + 1.3)
    box = (2, 3, PW - 4, h - 6)
    a, b = (np.asarray(hold['interval']) / 1000.0)
    X, _ = xy([a, b], [0, 0], box, xlim, ylim)
    _, ym = xy([0], [med], box, xlim, ylim)
    return svg(PW, h, [
        f'<rect x="{X[0]:.1f}" y="2" width="{X[1] - X[0]:.1f}" height="{h - 4}" fill="{AMBER}" opacity="0.10" rx="3"/>',
        band(d['hold_t'], d['hold_torque'], box, xlim, ylim, MUTED, 150, 0.9),
        hline(X[0], X[1], ym[0], AMBER, 2, '5 3'),
        text(X[0] + 6, ym[0] - 7, f'{hold.get("stat", "median")} {med:+.2f} Nm'.replace('-', '−'), 10.5, AMBER, weight=700),
    ])


def balance_plot(h, w):
    """New method: each angle's target divided by its session-1 swing (SD)."""
    d = D['test']
    now = int(np.argmin(abs(KNOTS - d['angle_now'])))
    bw, gap, top = 15, 6, 14
    bh = h - top - 13
    body = []
    for j, s in enumerate(SD):
        x, hh = j * (bw + gap), s / SD.max() * bh
        body.append(f'<rect x="{x}" y="{top + bh - hh:.1f}" width="{bw}" height="{hh:.1f}" rx="2" '
                    f'fill="{TEAL if j == now else "#C9D1DB"}"/>')
    xn = now * (bw + gap) + bw / 2
    body.append(text(xn, top + bh - SD[now] / SD.max() * bh - 3, f'{SD[now]:.1f} Nm', 10, TEAL, 'middle', 700))
    body.append(text(0, h - 1, 'swing per angle, toes down → up', 9, MUTED, weight=600))
    x0 = 8 * (bw + gap) + 30
    body.append(text(x0 - 15, h / 2 + 8, '÷', 24, TEAL, 'middle', 700))
    box = (x0 + 6, 16, w - x0 - 8, h - 20)
    xlim, ylim = SPAN['test'], (-2.1, 0.3)
    _, y0 = xy([0], [0], box, xlim, ylim)
    body.append(hline(box[0], w - 2, y0[0], LINE))
    body.append(path(d['t'], d['target'] / SD[now], box, xlim, ylim, TEAL, 2.2))
    body.append(text(box[0], 10, f'target ÷ {SD[now]:.1f} Nm: what this angle trains on', 10, TEAL, weight=700))
    return svg(w, h, body)


def map_plot(s, h):
    d = D[s]
    other = 'retest' if s == 'test' else 'test'
    color, ocolor = (S1, S2) if s == 'test' else (S2, S1)
    xlim, ylim = (-0.8, 0.32), (-11, 4)
    box = (24, 4, PW - 30, h - 10)
    body = []
    _, y0 = xy([0], [0], box, xlim, ylim)
    body.append(hline(box[0], PW - 4, y0[0], LINE))
    m, o = maps[s], maps[other]
    body.append(path(o.positions, o.torques, box, xlim, ylim, ocolor, 1.5, '3 2', 0.45))
    body.append(path(m.positions, m.torques, box, xlim, ylim, color, 2.2))
    X, Y = xy(m.positions, m.torques, box, xlim, ylim)
    for p, q in zip(X, Y):
        body.append(f'<circle cx="{p:.1f}" cy="{q:.1f}" r="2.8" fill="{color}"/>')
    hx, hy = xy([d['hold']['position']], [d['hold']['torque']], box, xlim, ylim)
    body.append(f'<circle cx="{hx[0]:.1f}" cy="{hy[0]:.1f}" r="5.5" fill="none" stroke="{AMBER}" stroke-width="1.6"/>')
    ax, ay = xy([d['angle_now']], [d['map_now']], box, xlim, ylim)
    body.append(f'<line x1="{ax[0]:.1f}" x2="{ax[0]:.1f}" y1="{box[1]}" y2="{box[1] + box[3]}" stroke="{INK}" stroke-width="1" stroke-dasharray="2 2"/>')
    body.append(f'<rect x="{ax[0] - 3.5:.1f}" y="{ay[0] - 3.5:.1f}" width="7" height="7" fill="{INK}" transform="rotate(45 {ax[0]:.1f} {ay[0]:.1f})"/>')
    body.append(text(ax[0] - 8, ay[0] + 16, f'{d["map_now"]:+.1f}'.replace('-', '−'), 10.5, INK, 'end', 700))
    body.append(text(0, box[1] + 9, 'Nm', 9.5, MUTED, weight=600))
    body.append(text(box[0], h - 1, 'toes down', 9.5, MUTED, weight=600))
    body.append(text(PW - 4, h - 1, 'up', 9.5, MUTED, 'end', 600))
    return svg(PW, h, body)


def score_plot(h):
    return trace_plot('retest', h, ('target', 'pred'), w=300)


# ----------------------------------------------------------------------------- layout
W, H = 1664, 700
COL = [136, 394, 652, 910, 1168, 1426]
CW = 236
ROWS = {'R1': (10, 128), 'R2': (166, 168), 'R3': (382, 154), 'R4': (564, 126)}
LANES = {'test': (0, 344, S1, '#EBF1F8', 'SESSION 1 · TEST · trains the model'),
         'retest': (372, 700, S2, '#FAEEE6', 'SESSION 2 · RETEST · scores it')}
TITLE_H, CAP_H, PAD = 22, 18, 8


def plot_h(row):
    return ROWS[row][1] - 2 * PAD - TITLE_H - CAP_H - 4


nodes = []


def node(col, row, title, plot, caption, span=1, kind='', tag=''):
    x, (y, h) = COL[col], ROWS[row]
    w = CW * span + 22 * (span - 1)
    tag_html = f'<span class="tag">{tag}</span>' if tag else ''
    nodes.append(f'<div class="node {kind}" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px">'
                 f'<div class="title">{title}{tag_html}</div><div class="plot">{plot}</div>'
                 f'<div class="cap">{caption}</div></div>')


p1, p2, p3, p4 = (plot_h(r) for r in ('R1', 'R2', 'R3', 'R4'))
NEWTAG = '<span class="newtag">NEW</span>'
STEPS = 20 if NEW else 50
HOLD_RULE = 'still ≥ 5 s, SD < 1 Nm → mean' if NEW else 'still ≥ 3 s, ±0.5 Nm → median'
# Session 1: the answer key (outer row) and the model path (inner row).
node(0, 'R1', 'Relaxed passive trial', hold_plot('test', p1), HOLD_RULE, kind='key')
node(1, 'R1', 'Passive map', map_plot('test', p1), 'same idea as last week' if NEW else '8 holds, read at the angle',
     kind='key map', tag='SESSION 1')
node(2, 'R1', 'Measured − passive', measured_plot('test', p1), 'subtract, every 10 ms', kind='key')
node(3, 'R1', 'Target: active torque', trace_plot('test', p1, ('target',)), 'what the muscles add', kind='key')
if NEW:
    node(4, 'R1', f'Balanced loss {NEWTAG}', balance_plot(p1, CW * 2 + 22 - 20),
         'each angle ÷ its session-1 swing → all 8 angles count the same', span=2, kind='loss new')
else:
    node(4, 'R1', 'Training: compare, adjust <span class="legend"><i class="sw t"></i>target <i class="sw p"></i>prediction</span>',
         trace_plot('test', p1, ('target', 'pred'), w=CW * 2 + 22 - 20), 'error → adjust weights · ≤ 60 epochs, early stop on the held-out 20%', span=2, kind='loss')
node(0, 'R2', 'Push trial · raw 1 kHz', raw_plot('test', p2), '80% trains · last 20% checks')
node(1, 'R2', 'Preprocessing (SOL)', stage_plot('test', p2), '→ 100 Hz · ÷ session-1 max')
if NEW:
    node(2, 'R2', f'Auto-gain {NEWTAG}', env_plot('test', p2), 'rescaled from its own past', kind='new')
    node(3, 'R2', f'0.2 s windows {NEWTAG}', window_plot('test', p2), 'every window trains', kind='new')
    node(4, 'R2', 'LSTM', lstm_plot(p2, steps=STEPS), '2 layers × 8 units → 1 value')
    nodes.append(
        f'<div class="node new budget" style="left:{COL[5]}px;top:{ROWS["R2"][0]}px;width:{CW}px;height:{ROWS["R2"][1]}px">'
        f'<div class="title">Training budget {NEWTAG}</div><div class="rows">'
        '<p><b>every</b> window<span>was every 5th</span></p><p>batch <b>8</b><span>was 64</span></p>'
        '<p>up to <b>200</b> epochs<span>was 60</span></p><p>stop after <b>25</b> flat<span>was 12</span></p></div>'
        '<div class="cap">same LSTM, trained longer</div></div>')
else:
    node(2, 'R2', 'Model input · 100 Hz', env_plot('test', p2), '4 envelopes + ankle angle')
    node(3, 'R2', 'Cut into 0.5 s windows', window_plot('test', p2), 'every 5th window trains')
    node(4, 'R2', 'LSTM', lstm_plot(p2), '2 layers × 8 units → 1 value')
    node(5, 'R2', 'Prediction', trace_plot('test', p2, ('pred',)), 'active torque at window end')
# Session 2: the model path (inner row) and the answer key (outer row).
node(0, 'R3', 'Push trial · retest', raw_plot('retest', p3), 'all 90 s · never trained on')
node(1, 'R3', 'Same preprocessing', stage_plot('retest', p3), '→ 100 Hz · ÷ the session-1 max')
if NEW:
    node(2, 'R3', f'Auto-gain, live {NEWTAG}', env_plot('retest', p3), 'adapts to session 2’s own past', kind='new')
    node(3, 'R3', f'Every 0.2 s window {NEWTAG}', window_plot('retest', p3), 'every window, 10 ms apart', kind='new')
else:
    node(2, 'R3', 'Model input · 100 Hz', env_plot('retest', p3), '4 envelopes + ankle angle')
    node(3, 'R3', 'Every 0.5 s window', window_plot('retest', p3), 'every window, 10 ms apart')
node(4, 'R3', 'Same LSTM · frozen', lstm_plot(p3, frozen=True, steps=STEPS), 'weights fixed after session 1', kind='frozen')
node(5, 'R3', 'Prediction', trace_plot('retest', p3, ('pred',)),
     '× the angle’s swing → Nm' if NEW else 'active torque, every 10 ms')
node(0, 'R4', 'Relaxed passive trial', hold_plot('retest', p4), 'same rule, session 2 holds', kind='key')
node(1, 'R4', 'Passive map', map_plot('retest', p4), 'the retest map', kind='key map s2', tag='SESSION 2')
node(2, 'R4', 'Measured − passive', measured_plot('retest', p4), 'subtract, every 10 ms', kind='key')
node(3, 'R4', 'Target: active torque', trace_plot('retest', p4, ('target',)), 'the answer key', kind='key')
r2, mae = score['r2'], score['mae']
was = (f'<p class="was">last week {OLD_SCORE["r2"]:.2f} · {OLD_SCORE["mae"]:.1f} Nm</p>' if NEW else '')
nodes.append(
    f'<div class="node score" style="left:{COL[4]}px;top:{ROWS["R4"][0]}px;width:{CW * 2 + 22}px;height:{ROWS["R4"][1]}px">'
    f'<div class="title">Score at this angle <span class="legend"><i class="sw t"></i>target <i class="sw p"></i>prediction</span></div>'
    f'<div class="scorebody"><div class="plot">{score_plot(p4)}</div><div class="numcol">'
    f'<div class="nums"><div><b>{r2:.2f}</b><span>R²</span></div><div><b>{mae:.1f}</b><span>MAE, Nm</span></div></div>{was}</div></div>'
    f'<div class="cap">same for all 8 angles · 6 people × 10 seeds</div></div>')


# ----------------------------------------------------------------------------- connectors
def mid(col, row, side, span=1, dx=0):
    x, (y, h) = COL[col], ROWS[row]
    w = CW * span + 22 * (span - 1)
    return {'l': (x, y + h / 2), 'r': (x + w, y + h / 2), 't': (x + w / 2 + dx, y), 'b': (x + w / 2 + dx, y + h)}[side]


def arrow(points, color=INK, width=1.6, dash=None, marker='a'):
    d = 'M' + ' L'.join(f'{x:.1f} {y:.1f}' for x, y in points)
    extra = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"'
            f'{extra} marker-end="url(#{marker})"/>')


def short(p, q, gap=3):
    """Pull the end point back so the arrow head's tip meets the box edge."""
    (x0, y0), (x1, y1) = p, q
    L = np.hypot(x1 - x0, y1 - y0)
    return (x1 - (x1 - x0) / L * gap, y1 - (y1 - y0) / L * gap)


links = []
for row, color in (('R2', INK), ('R3', INK), ('R1', AMBER), ('R4', AMBER)):
    last = (4 if NEW and row == 'R2' else 5) if row in ('R2', 'R3') else 3
    for c in range(last):
        a, b = mid(c, row, 'r'), mid(c + 1, row, 'l')
        links.append(arrow([a, short(a, b)], color, 1.8, marker='a' if color == INK else 'k'))
# Target into the comparison, prediction into the comparison.
a, b = mid(3, 'R1', 'r'), mid(4, 'R1', 'l')
links.append(arrow([a, short(a, b)], AMBER, 1.8, marker='k'))
a, b = mid(3, 'R4', 'r'), mid(4, 'R4', 'l')
links.append(arrow([a, short(a, b)], AMBER, 1.8, marker='k'))
a = mid(5, 'R3', 'b')
links.append(arrow([a, short(a, (a[0], ROWS['R4'][0]))], INK, 1.8))
x = COL[4] + CW / 2
r1b, r2t = ROWS['R1'][0] + ROWS['R1'][1], ROWS['R2'][0]
if NEW:
    # Training loop: the LSTM's output goes up into the balanced loss, the error comes back down.
    xu, xd = COL[4] + 56, COL[4] + CW - 56
    links.append(arrow([(xu, r2t), short((xu, r2t), (xu, r1b))], INK, 1.8))
    links.append(arrow([(xd, r1b), short((xd, r1b), (xd, r2t))], TEAL, 2.4, marker='g'))
else:
    a = mid(5, 'R2', 't')
    links.append(arrow([a, short(a, (a[0], r1b))], INK, 1.8))
    # Training loop: the error goes back into the LSTM.
    links.append(arrow([(x, r1b), short((x, 150), (x, r2t))], TEAL, 2.4, marker='g'))
# Measured torque from the push trial to the subtraction, in the gap between the rows.
xs, xm = COL[0] + CW - 36, COL[2] + CW / 2
gy1 = (ROWS['R1'][0] + ROWS['R1'][1] + ROWS['R2'][0]) / 2 + 4
links.append(arrow([(xs, ROWS['R2'][0]), (xs, gy1), (xm, gy1), short((xm, gy1), (xm, ROWS['R1'][0] + ROWS['R1'][1]))], MUTED, 1.6, marker='m'))
gy2 = (ROWS['R3'][0] + ROWS['R3'][1] + ROWS['R4'][0]) / 2 - 4
links.append(arrow([(xs, ROWS['R3'][0] + ROWS['R3'][1]), (xs, gy2), (xm, gy2), short((xm, gy2), (xm, ROWS['R4'][0]))], MUTED, 1.6, marker='m'))
# Trained weights travel from session 1 to session 2.
links.append(arrow([(x, ROWS['R2'][0] + ROWS['R2'][1]), short((x, 340), (x, ROWS['R3'][0]))], INK, 3, '6 4', marker='w'))

labels = [
    (f'<div class="lbl" style="left:{xu - 106}px;top:{r1b + 6}px;width:98px;text-align:right;color:{INK}">ŷ (÷ swing)</div>'
     f'<div class="lbl" style="left:{xd + 8}px;top:{r1b + 6}px;width:118px;color:{TEAL}">adjust weights</div>') if NEW else
    f'<div class="lbl" style="left:{x - 128}px;top:{ROWS["R1"][0] + ROWS["R1"][1] + 3}px;width:118px;text-align:right;color:{TEAL}">adjust weights</div>',
    f'<div class="lbl" style="left:{x + 12}px;top:{ROWS["R2"][0] + ROWS["R2"][1] + 13}px;width:220px;color:{INK}">trained weights, frozen</div>',
    f'<div class="lbl" style="left:{xs + 10}px;top:{gy1 - 17}px;width:200px;color:{MUTED}">measured torque</div>',
    f'<div class="lbl" style="left:{xs + 10}px;top:{gy2 + 3}px;width:200px;color:{MUTED}">measured torque</div>',
]

# Lanes, tabs and the tree's branches.
back = []
for s, (y0, y1, color, tint, name) in LANES.items():
    back.append(f'<rect x="46" y="{y0}" width="{W - 46}" height="{y1 - y0}" rx="16" fill="{tint}"/>')
    back.append(f'<rect x="46" y="{y0}" width="30" height="{y1 - y0}" rx="10" fill="{color}"/>')
    cy = (y0 + y1) / 2
    rows = ('R1', 'R2') if s == 'test' else ('R3', 'R4')
    for r in rows:
        ry = ROWS[r][0] + ROWS[r][1] / 2
        back.append(f'<path d="M76 {cy:.1f} C 108 {cy:.1f}, 100 {ry:.1f}, 132 {ry:.1f}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        back.append(f'<circle cx="133" cy="{ry:.1f}" r="3.5" fill="{color}"/>')
    back.append(f'<path d="M20 350 C 34 350, 30 {cy:.1f}, 46 {cy:.1f}" fill="none" stroke="{color}" stroke-width="3.5" stroke-linecap="round"/>')
back.append(f'<circle cx="18" cy="350" r="17" fill="{INK}"/>')
tabs = [f'<div class="tab" style="top:{y0}px;height:{y1 - y0}px">{name}</div>' for y0, y1, _, _, name in LANES.values()]

MARKERS = ''.join(
    f'<marker id="{k}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="{sz}" markerHeight="{sz}" markerUnits="userSpaceOnUse" orient="auto">'
    f'<path d="M0 0 L8 4 L0 8 Z" fill="{c}"/></marker>'
    for k, c, sz in (('a', INK, 9), ('k', AMBER, 9), ('g', TEAL, 11), ('m', MUTED, 9), ('w', INK, 13)))

html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pipeline tree</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700;800&display=swap">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ width: {W}px; height: {H}px; background: #F7F5F0; overflow: hidden; }}
body {{ position: relative; font-family: 'Public Sans', Arial, sans-serif; color: {INK}; }}
svg text {{ font-family: 'Public Sans', Arial, sans-serif; }}
.layer {{ position: absolute; left: 0; top: 0; }}
.node {{ position: absolute; background: #FFFDF9; border: 1px solid {LINE}; border-radius: 10px; padding: {PAD}px 10px;
        display: flex; flex-direction: column; gap: 2px; box-shadow: 0 1px 3px rgba(27,36,48,.07); }}
.node .title {{ height: {TITLE_H}px; font-size: 15.5px; font-weight: 700; line-height: {TITLE_H}px; white-space: nowrap;
               display: flex; align-items: center; gap: 8px; }}
.node .plot {{ flex: 1; display: flex; align-items: center; }}
.node .cap {{ height: {CAP_H}px; font-size: 13px; font-weight: 500; color: #4A5568; line-height: {CAP_H}px; white-space: nowrap; }}
.node.key {{ border-color: #E2C7A6; background: #FFFCF6; }}
.node.map {{ border: 2px solid {S1}; box-shadow: 0 0 0 4px rgba(47,109,181,.12); }}
.node.map.s2 {{ border-color: {S2}; box-shadow: 0 0 0 4px rgba(194,65,12,.12); }}
.node.loss {{ border: 1.5px solid {TEAL}; }}
.node.score {{ border: 2px solid {INK}; }}
.node.frozen {{ background: #F7F8FA; }}
.node.new {{ border: 2px solid {TEAL}; box-shadow: 0 0 0 4px rgba(31,122,114,.14); }}
.newtag {{ font-size: 10.5px; font-weight: 800; letter-spacing: 1px; color: #fff; background: {TEAL}; border-radius: 4px; padding: 1px 6px; line-height: 16px; }}
.budget .rows {{ flex: 1; display: flex; flex-direction: column; justify-content: center; gap: 5px; }}
.budget .rows p {{ font-size: 14px; font-weight: 500; display: flex; justify-content: space-between; align-items: baseline; gap: 6px;
                  border-bottom: 1px solid #E6E1D8; padding-bottom: 3px; }}
.budget .rows b {{ font-weight: 800; color: {TEAL}; }}
.budget .rows span {{ margin-left: auto; font-size: 12px; font-weight: 600; color: {MUTED}; }}
.numcol {{ display: flex; flex-direction: column; gap: 4px; }}
.was {{ font-size: 12px; font-weight: 600; color: {MUTED}; }}
.tag {{ font-size: 11px; font-weight: 800; letter-spacing: 1px; color: #fff; background: {S1}; border-radius: 4px; padding: 1px 6px; line-height: 16px; }}
.s2 .tag {{ background: {S2}; }}
.legend {{ margin-left: auto; font-size: 12px; font-weight: 600; color: {MUTED}; display: flex; align-items: center; gap: 5px; }}
.sw {{ display: inline-block; width: 18px; height: 0; border-top: 2.4px solid {TEAL}; margin-left: 6px; }}
.sw.p {{ border-top: 2px dashed {INK}; }}
.scorebody {{ flex: 1; display: flex; gap: 14px; align-items: center; }}
.nums {{ display: flex; gap: 18px; }}
.nums div {{ display: flex; flex-direction: column; align-items: flex-start; }}
.nums b {{ font-size: 30px; font-weight: 800; line-height: 32px; color: {INK}; }}
.nums span {{ font-size: 12px; font-weight: 600; color: {MUTED}; }}
.tab {{ position: absolute; left: 46px; width: 30px; writing-mode: vertical-rl; transform: rotate(180deg); color: #fff;
       font-size: 12.5px; font-weight: 800; letter-spacing: 1.6px; text-align: center; line-height: 30px; white-space: nowrap; }}
.root {{ position: absolute; left: 1px; top: 333px; width: 34px; height: 34px; color: #fff; font-size: 12px; font-weight: 800;
        text-align: center; line-height: 34px; }}
.lbl {{ position: absolute; font-size: 12.5px; font-weight: 700; line-height: 14px; white-space: nowrap; }}
</style></head><body>
<svg class="layer" width="{W}" height="{H}" viewBox="0 0 {W} {H}">{''.join(back)}</svg>
{''.join(tabs)}
<div class="root">{SUBJECT}</div>
{''.join(nodes)}
<svg class="layer" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs>{MARKERS}</defs>{''.join(links)}</svg>
{''.join(labels)}
</body></html>
"""
Path(out).write_text(html, encoding='utf-8')
print('wrote', out, f'{len(html) / 1024:.0f} KB')
