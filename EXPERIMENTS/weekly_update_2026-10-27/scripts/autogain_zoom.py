"""Zoom on auto-gain for the weekly-update deck: what it does to one muscle, step by step.

Real data: JM at -0.36 rad, soleus, session 1 vs session 2, computed by the new method's own
code (new_method_data.py ... JM -0.36). Writes a 1664 x 700 HTML page; render it with headless
Edge like the pipeline tree.

Usage:
    python autogain_zoom.py <new_method_JM.npz> <output.html>
"""
import sys
from pathlib import Path

import numpy as np

src, out = sys.argv[1:3]
z = np.load(src)
K, NAME = 2, 'SOL'                       # soleus
INK, MUTED, LINE = '#1B2430', '#5B6573', '#D9D3C7'
S1, S2, TEAL, AMBER = '#2F6DB5', '#C2410C', '#1F7A72', '#B45309'
W, H = 1664, 700


def size(env, tq):
    swing = np.percentile(tq, 95) - np.percentile(tq, 5)
    return (np.percentile(env, 95, axis=0) - np.percentile(env, 10, axis=0)) / swing


before = size(z['S2_base'], z['S2_tq']) / size(z['S1_base'], z['S1_tq'])
after = size(z['S2_agc'], z['S2_tq']) / size(z['S1_agc'], z['S1_tq'])
print('S2/S1 size per Nm before', np.round(before, 2), 'after', np.round(after, 2))

# The moment used for the worked example: mid-push in session 2, once the 45 s look-back is full.
t2, x2, lo2, top2 = z['S2_t'], z['S2_base'][:, K], z['S2_lo'][:, K], z['S2_top'][:, K]
frac = (x2 - lo2) / (top2 - lo2)
cand = np.where((t2 > 55) & (t2 < 75))[0]
TI = cand[np.argmin(abs(frac[cand] - 0.75))]
TS = float(t2[TI])
xv, fv, tv, ov = x2[TI], lo2[TI], top2[TI], z['S2_agc'][TI, K]
print(f't* {TS:.2f} s: x {xv:.4f}, floor {fv:.4f}, top {tv:.4f} -> {(xv - fv) / (tv - fv):.3f} (model input {ov:.3f})')


# ----------------------------------------------------------------------------- svg helpers
def xy(t, y, box, xlim, ylim):
    x0, y0, w, h = box
    X = x0 + (np.asarray(t, float) - xlim[0]) / (xlim[1] - xlim[0]) * w
    Y = y0 + h - (np.clip(y, *ylim) - ylim[0]) / (ylim[1] - ylim[0]) * h
    return X, Y


def path(t, y, box, xlim, ylim, color, width=1.6, dash=None, opacity=1):
    X, Y = xy(t, y, box, xlim, ylim)
    d = 'M' + ' L'.join(f'{p:.1f} {q:.1f}' for p, q in zip(X, Y))
    extra = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" '
            f'opacity="{opacity}"{extra}/>')


def text(x, y, s, size=13, color=MUTED, anchor='start', weight=600):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def hline(x0, x1, y, color=LINE, width=1, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x0:.1f}" x2="{x1:.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="{width}"{extra}/>'


def svg(w, h, body):
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">' + ''.join(body) + '</svg>'


def pair(key, ylim, h, guide=None):
    """Session 1 | session 2 side by side: the muscle on one shared scale, torque below."""
    w, gap = 360, 36
    th = 40
    eh = h - 20 - 8 - th - 20
    body = []
    tlim = (min(z['S1_tq'].min(), z['S2_tq'].min()) - 0.3, max(z['S1_tq'].max(), z['S2_tq'].max()) + 0.3)
    for j, (s, color, name) in enumerate((('S1', S1, 'Session 1'), ('S2', S2, 'Session 2'))):
        x0 = j * (w + gap)
        t = z[f'{s}_t']
        xlim = (t[0], t[-1])
        body.append(text(x0, 13, name, 13.5, color, weight=800))
        box = (x0, 20, w, eh)
        body.append(hline(x0, x0 + w, 20 + eh, LINE))
        if guide is not None:
            _, yg = xy([0], [guide], box, xlim, ylim)
            body.append(hline(x0, x0 + w, yg[0], TEAL, 1.2, '5 4'))
        body.append(path(t[::2], z[f'{s}_{key}'][::2, K], box, xlim, ylim, color, 1.4))
        tbox = (x0, 20 + eh + 8, w, th)
        body.append(path(t[::4], z[f'{s}_tq'][::4], tbox, xlim, tlim, MUTED, 1.2))
        body.append(text(x0, h - 3, f'torque · same pushes ({np.percentile(z[f"{s}_tq"], 95) - np.percentile(z[f"{s}_tq"], 5):.1f} Nm swing)', 11.5, MUTED))
    return svg(2 * w + gap, h, body)


# ----------------------------------------------------------------------------- cards
p_lim = (0, max(np.percentile(z['S1_base'][:, K], 99.8), np.percentile(z['S2_base'][:, K], 99.8)) * 1.08)
card1 = pair('base', p_lim, 236)
card4 = pair('agc', (0, 1.6), 236, guide=1.0)


def levels(h):
    w = 760
    t, xlim = t2, (0, t2[-1])
    ylim = (0, np.percentile(x2, 99.8) * 1.15)
    box = (0, 26, w, h - 50)
    body = []
    # look-back windows at t*
    for span, color, y, lab in ((45, TEAL, 19, 'past 45 s → top = 95th percentile'),
                                (20, INK, h - 12, 'past 20 s → floor = 10th percentile')):
        a, _ = xy([max(TS - span, 0)], [0], box, xlim, ylim)
        b, _ = xy([TS], [0], box, xlim, ylim)
        body.append(f'<rect x="{a[0]:.1f}" y="{box[1]}" width="{b[0] - a[0]:.1f}" height="{box[3]}" fill="{color}" opacity="0.07"/>')
        yy = y if span == 45 else box[1] + box[3] + 8
        body.append(f'<path d="M{a[0]:.1f} {yy + 4} V{yy} H{b[0]:.1f} V{yy + 4}" fill="none" stroke="{color}" stroke-width="1.4"/>' if span == 45 else
                    f'<path d="M{a[0]:.1f} {yy - 4} V{yy} H{b[0]:.1f} V{yy - 4}" fill="none" stroke="{color}" stroke-width="1.4"/>')
        body.append(text(a[0] + 4, yy - 5 if span == 45 else yy + 14, lab, 12, color, weight=700))
    body.append(path(t[::2], x2[::2], box, xlim, ylim, S2, 1.2, opacity=0.85))
    body.append(path(t, top2, box, xlim, ylim, TEAL, 2.4))
    body.append(path(t, lo2, box, xlim, ylim, INK, 2.2, '5 3'))
    X, Y = xy([TS], [xv], box, xlim, ylim)
    body.append(f'<line x1="{X[0]:.1f}" x2="{X[0]:.1f}" y1="{box[1]}" y2="{box[1] + box[3]}" stroke="{INK}" stroke-width="1.2"/>')
    body.append(f'<circle cx="{X[0]:.1f}" cy="{Y[0]:.1f}" r="5" fill="{S2}" stroke="#fff" stroke-width="1.5"/>')
    body.append(text(X[0] + 7, box[1] + 12, 'now', 12, INK, weight=800))
    _, yt = xy([0], [top2[-1]], box, xlim, ylim)
    _, yf = xy([0], [lo2[-1]], box, xlim, ylim)
    body.append(text(w, yt[0] - 6, 'top', 12.5, TEAL, 'end', 800))
    body.append(text(w, yf[0] - 6, 'floor', 12.5, INK, 'end', 800))
    return svg(w, h, body)


def gauge(h):
    w = 300
    yb, yt = h - 30, 34          # 0 and 1 on both scales
    left, right = 92, 250
    sc = lambda v: yb - (v - fv) / (tv - fv) * (yb - yt)
    body = [
        text(left, 16, 'session-2 signal', 12, S2, 'middle', 800),
        text(right, 16, 'model input', 12, TEAL, 'middle', 800),
        f'<line x1="{left}" x2="{left}" y1="{yt - 6}" y2="{yb + 6}" stroke="{MUTED}" stroke-width="2"/>',
        f'<line x1="{right}" x2="{right}" y1="{yt - 6}" y2="{yb + 6}" stroke="{MUTED}" stroke-width="2"/>',
    ]
    for v, lab, out_lab, color in ((tv, f'top {tv:.3f}', '1', TEAL), (fv, f'floor {fv:.3f}', '0', INK),
                                   (xv, f'now {xv:.3f}', f'{(xv - fv) / (tv - fv):.2f}', S2)):
        y = sc(v)
        body.append(f'<line x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="{2.2 if color == S2 else 1.4}"'
                    f'{" stroke-dasharray=" + chr(34) + "4 3" + chr(34) if color != S2 else ""}/>')
        body.append(f'<circle cx="{left}" cy="{y:.1f}" r="4.5" fill="{color}"/><circle cx="{right}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
        body.append(text(left - 10, y + 4, lab, 12, color, 'end', 700))
        body.append(text(right + 10, y + 5, out_lab, 15 if color == S2 else 13, color, weight=800))
    return svg(w, h, body)


cards = [
    ('c1', 0, 0, 800, 330, '1', 'Last week: ÷ the session-1 maximum',
     card1, f'<b class="big s2">{before[K]:.2f}×</b>as big in session 2 for the same push',
     'JM, soleus, −0.36 rad. A fixed scale reads the weaker signal as a weaker push.'),
    ('c2', 864, 0, 800, 330, '2', 'Auto-gain tracks two running levels, from the past only',
     levels(236), '', 'Recomputed every 0.5 s from past samples; the first seconds lean on session-1 values.'),
    ('c3', 0, 370, 520, 330, '3', 'Rescale every sample',
     gauge(230), '<b class=why>Why it works</b>A quieter session has a lower floor and top too, so the ratio, and what the model sees, stays the same.',
     '(now − floor) ÷ (top − floor), kept between 0 and 2'),
    ('c4', 584, 370, 1080, 330, '4', 'After auto-gain: one scale for both sessions',
     card4, f'<b class="big teal">{after[K]:.2f}×</b>as big for the same push<span class="rows">' + '<br>'.join(f'{m} {before[i]:.2f}× → {after[i]:.2f}×' for i, m in enumerate(('GM', 'GL', 'SOL'))) + '</span>',
     'Dashed line = the running top. The LSTM now sees session 2 at the size it learned from.'),
]
html_cards = []
for cid, x, y, w, h, num, title, plot, badge, cap in cards:
    badge_cls = 'badge float' if cid == 'c1' else 'badge'
    badge_html = f'<div class="{badge_cls}">{badge}</div>' if badge else ''
    html_cards.append(
        f'<div class="card {cid}" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px">'
        f'<div class="title"><span class="num">{num}</span>{title}</div>'
        f'<div class="body"><div class="plot">{plot}</div>{badge_html}</div>'
        f'<div class="cap">{cap}</div></div>')

arrows = []
mk = (f'<marker id="a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="12" markerHeight="12" markerUnits="userSpaceOnUse" '
      f'orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{TEAL}"/></marker>')
arrows.append(f'<path d="M802 165 H858" stroke="{TEAL}" stroke-width="3" fill="none" marker-end="url(#a)"/>')
arrows.append(f'<path d="M1264 332 V350 H260 V364" stroke="{TEAL}" stroke-width="3" fill="none" stroke-linejoin="round" marker-end="url(#a)"/>')
arrows.append(f'<path d="M522 535 H578" stroke="{TEAL}" stroke-width="3" fill="none" marker-end="url(#a)"/>')

html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Auto-gain zoom</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700;800&display=swap">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ width: {W}px; height: {H}px; background: #F7F5F0; overflow: hidden; }}
body {{ position: relative; font-family: 'Public Sans', Arial, sans-serif; color: {INK}; }}
svg text {{ font-family: 'Public Sans', Arial, sans-serif; }}
.card {{ position: absolute; background: #FFFDF9; border: 1px solid {LINE}; border-radius: 12px; padding: 14px 18px;
        display: flex; flex-direction: column; gap: 8px; box-shadow: 0 1px 3px rgba(27,36,48,.07); }}
.c2, .c3 {{ border: 2px solid {TEAL}; box-shadow: 0 0 0 4px rgba(31,122,114,.12); }}
.title {{ font-size: 18px; font-weight: 800; display: flex; align-items: center; gap: 10px; white-space: nowrap; }}
.num {{ display: inline-flex; width: 26px; height: 26px; border-radius: 50%; background: {TEAL}; color: #fff; font-size: 14px;
       align-items: center; justify-content: center; }}
.c1 .num {{ background: {MUTED}; }}
.body {{ flex: 1; display: flex; gap: 18px; align-items: center; position: relative; }}
.plot {{ display: flex; }}
.badge {{ font-size: 13px; font-weight: 600; color: {MUTED}; line-height: 1.45; display: flex; flex-direction: column; gap: 4px; max-width: 220px; }}
.badge.float {{ position: absolute; right: 6px; top: 26px; text-align: right; align-items: flex-end; max-width: 230px; }}
.rows {{ margin-top: 8px; font-size: 13.5px; font-weight: 700; color: {INK}; line-height: 1.6; }}
.why {{ font-size: 14px; font-weight: 800; color: {TEAL}; }}
.big {{ font-size: 30px; font-weight: 800; line-height: 32px; }}
.big.s2 {{ color: {S2}; }}
.big.teal {{ color: {TEAL}; }}
.cap {{ font-size: 13.5px; font-weight: 500; color: #4A5568; white-space: nowrap; }}
.formula {{ position: absolute; right: 18px; top: 74px; width: 150px; font-size: 13px; font-weight: 600; color: #4A5568; line-height: 1.45; }}
.formula b {{ color: {INK}; }}
</style></head><body>
{''.join(html_cards)}
<svg style="position:absolute;left:0;top:0" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs>{mk}</defs>{''.join(arrows)}</svg>
</body></html>
"""
Path(out).write_text(html, encoding='utf-8')
print('wrote', out)
