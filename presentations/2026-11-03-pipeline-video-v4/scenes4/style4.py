"""Shared look and helpers for the v2 chapter scenes (no code on screen).

Scenes only READ data: data4/*.npz and data4/facts_*.json (v2 extraction), data/*.npz and data/facts_*.json
(v1 extraction, reused where the baseline and D01 share a step), and audio4/beats.json (beat durations).
Narration text lives in narration4/narration.json (built from STORYBOARD2.md).

Gotchas (manim 0.22, see README): Text drops leading spaces; Axes draws its x axis at y = 0 (use Plot);
VGroup cannot hold a ValueTracker (use Group); there is no GrowFromCorner; Consolas has no ⊙.
"""
import json
import textwrap
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, ORIGIN, RIGHT, UL, UP, UR, Axes, DashedVMobject, FadeIn, FadeOut, Line,
                   MarkupText, Rectangle, ReplacementTransform, RoundedRectangle, Scene, Sector, Text, VGroup, VMobject,
                   config)

VIDEO = Path(__file__).resolve().parent.parent
DATA, DATA2, AUDIO = VIDEO / 'data4', VIDEO / 'data4', VIDEO / 'audio4'
NARR = json.loads((VIDEO / 'narration4' / 'narration.json').read_text(encoding='utf-8'))
BEATS = json.loads((AUDIO / 'beats.json').read_text(encoding='utf-8')) if (AUDIO / 'beats.json').exists() else {}
GAP = NARR['gap_s']
SAY = {b['id']: b['text'] for b in NARR['beats']}
VIS = {b['id']: b['vis'] for b in NARR['beats']}
CHAPTER_TITLE = {c['id']: c['title'] for c in NARR['chapters']}

# ── palette (same as v1) ─────────────────────────────────────────────────────
BG, INK, DIM, FAINT, PANEL = '#0E1116', '#E6EDF3', '#8B949E', '#30363D', '#161B22'
C_GM, C_GL, C_SOL, C_TA = '#2EC4B6', '#A78BFA', '#FFC857', '#F15BB5'
MUSCLE = {'GM': C_GM, 'GL': C_GL, 'SOL': C_SOL, 'TA': C_TA}
MUSCLE_C = [C_GM, C_GL, C_SOL, C_TA]
C_ANGLE, C_S1, C_S2 = '#C9D1D9', '#4C9AFF', '#FF7A45'      # angle trace, session 1, session 2
C_TQ, C_PASSIVE, C_PRED, C_GOOD, C_BAD, C_HI = '#FFFFFF', '#8B949E', '#9BE564', '#3FB950', '#F85149', '#F0E442'
C_FIT, C_VAL, C_TEST = C_S1, '#F0E442', C_S2               # fit / validation / test sets
ANGLE_C = ['#8E6CEF', '#6C8CF5', '#4FA8E8', '#3EC1C9', '#4FD39B', '#8BDD6A', '#C9DE4B', '#F5D547']  # toes down → up
FONT, MONO = 'Segoe UI', 'Consolas'
config.background_color = BG

TOP, BOTTOM = 3.3, -3.6        # usable y range under the chapter tag (frame is ±4 × ±7.11)


def T(s, size=26, color=INK, weight='NORMAL', font=FONT, **kw):
    return Text(str(s), font=font, font_size=size, color=color, weight=weight, **kw)


def M(s, size=26, color=INK, font=FONT, **kw):
    """Pango markup: <b>, <i>, <span foreground="#hex">…</span>. Escape & < > yourself."""
    return MarkupText(str(s), font=font, font_size=size, color=color, **kw)


CHAR_W = 0.0062        # Segoe UI: average glyph width in frame units per point of font size


def chars_for(width, size):
    """How many characters of Segoe UI at ``size`` fit in ``width`` frame units (for W's ``width``)."""
    return max(4, int(width / (CHAR_W * size)))


def W(s, size=26, color=INK, width=40, line_spacing=0.8, align=LEFT, **kw):
    """Wrapped plain text, ``width`` in characters (see chars_for). Left-aligned lines with even spacing;
    align=ORIGIN centres each line (short labels)."""
    lines = textwrap.wrap(str(s), width) or ['']
    if align is ORIGIN and len(lines) > 1:
        return VGroup(*[T(l, size, color, **kw) for l in lines]).arrange(DOWN, buff=0.1 * size / 26)
    return T('\n'.join(lines), size, color, line_spacing=line_spacing, **kw)


def chapter_tag(cid):
    """Top-left 'NN  Title' tag for chapter id 'cNN'."""
    n = int(cid[1:])
    tag = VGroup(T(f'{n:02d}', 22, C_HI, weight='BOLD'), T(CHAPTER_TITLE[cid], 22, DIM)).arrange(RIGHT, buff=0.18)
    return tag.to_corner(UL, buff=0.32)


def card(lines, width=6.0, title=None, color=FAINT, size=24, title_color=C_HI, fill=PANEL, chars=None, pad=0.3):
    """Rounded panel with an optional title and wrapped lines (str or Mobject)."""
    chars = chars or chars_for(width - 2 * pad, size)
    body = VGroup()
    for l in lines:
        body.add(l if not isinstance(l, str) else W(l, size, INK, width=chars))
    body.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
    if title:
        head = T(title, size + 2, title_color, weight='BOLD')
        content = VGroup(head, body).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
    else:
        content = body
    box = RoundedRectangle(corner_radius=0.14, width=max(width, content.width + 2 * pad),
                           height=content.height + 2 * pad, fill_color=fill, fill_opacity=1,
                           stroke_color=color, stroke_width=2)
    content.move_to(box.get_center()).align_to(box, LEFT).shift(RIGHT * pad)
    g = VGroup(box, content)
    g.box, g.content = box, content
    return g


def bullets(items, size=26, color=INK, chars=46, bullet='•', buff=0.24, bcolor=C_HI):
    rows = VGroup()
    for it in items:
        b = T(bullet, size, bcolor)
        txt = it if not isinstance(it, str) else W(it, size, color, width=chars)
        row = VGroup(b, txt).arrange(RIGHT, buff=0.2, aligned_edge=UP)
        rows.add(row)
    return rows.arrange(DOWN, buff=buff, aligned_edge=LEFT)


def fmt(v, nd=2, signed=False):
    """Number for the screen: unicode minus, fixed decimals."""
    s = f'{v:+.{nd}f}' if signed else f'{v:.{nd}f}'
    return s.replace('-', '−')


# ── plots without LaTeX: invisible Axes for mapping + our own frame ──────────
class Plot(VGroup):
    def __init__(self, x_range, y_range, width, height, xticks=(), yticks=(), xlabel='', ylabel='', tick_size=16,
                 axis_color=FAINT, xfmt=None, yfmt=None, grid=False):
        super().__init__()
        self.ax = Axes(x_range=[x_range[0], x_range[1], (x_range[1] - x_range[0])], y_range=[y_range[0], y_range[1],
                       (y_range[1] - y_range[0])], x_length=width, y_length=height, tips=False,
                       axis_config=dict(include_numbers=False, include_ticks=False, stroke_opacity=0, stroke_width=0))
        x0, x1, y0, y1 = x_range[0], x_range[1], y_range[0], y_range[1]
        self.x_range, self.y_range = (x0, x1), (y0, y1)
        self.frame = VGroup(Line(self.ax.c2p(x0, y0), self.ax.c2p(x1, y0), stroke_color=axis_color, stroke_width=1.6),
                            Line(self.ax.c2p(x0, y0), self.ax.c2p(x0, y1), stroke_color=axis_color, stroke_width=1.6))
        self.add(self.ax, self.frame)
        fx = xfmt or (lambda v: f'{v:g}')
        fy = yfmt or (lambda v: f'{v:g}')
        self.ticks = VGroup()
        for v in xticks:
            p = self.ax.c2p(v, y_range[0])
            self.ticks.add(Line(p, p + DOWN * 0.07, stroke_color=axis_color, stroke_width=1.5),
                           T(fx(v).replace('-', '−'), tick_size, DIM).next_to(p, DOWN, buff=0.1))
        for v in yticks:
            p = self.ax.c2p(x_range[0], v)
            self.ticks.add(Line(p, p + LEFT * 0.07, stroke_color=axis_color, stroke_width=1.5),
                           T(fy(v).replace('-', '−'), tick_size, DIM).next_to(p, LEFT, buff=0.1))
            if grid:
                self.ticks.add(DashedVMobject(Line(p, self.ax.c2p(x_range[1], v), stroke_color=FAINT, stroke_width=1),
                                              num_dashes=40))
        self.add(self.ticks)
        if xlabel:
            self.xlabel = T(xlabel, tick_size, DIM).next_to(self.frame[0], DOWN, buff=0.42 if len(xticks) else 0.15)
            self.add(self.xlabel)
        if ylabel:
            self.ylabel = T(ylabel, tick_size, DIM).rotate(np.pi / 2).next_to(self.frame[1], LEFT,
                                                                              buff=0.62 if len(yticks) else 0.15)
            self.add(self.ylabel)

    def p(self, x, y):
        return self.ax.c2p(x, y)

    def curve(self, x, y, color=INK, width=2.0, max_points=1500, clip=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if len(x) > max_points:
            idx = np.linspace(0, len(x) - 1, max_points).astype(int)
            x, y = x[idx], y[idx]
        if clip is None:
            clip = self.y_range
        y = np.clip(y, *clip)
        pts = [self.ax.c2p(a, b) for a, b in zip(x, y)]
        return VMobject(stroke_color=color, stroke_width=width).set_points_as_corners(pts)

    def hline(self, y, color=DIM, width=1.5, dashed=True, x0=None, x1=None):
        a = self.p(self.x_range[0] if x0 is None else x0, y)
        b = self.p(self.x_range[1] if x1 is None else x1, y)
        ln = Line(a, b, stroke_color=color, stroke_width=width)
        return DashedVMobject(ln, num_dashes=50) if dashed else ln

    def vline(self, x, color=DIM, width=1.5, dashed=True, y0=None, y1=None):
        a = self.p(x, self.y_range[0] if y0 is None else y0)
        b = self.p(x, self.y_range[1] if y1 is None else y1)
        ln = Line(a, b, stroke_color=color, stroke_width=width)
        return DashedVMobject(ln, num_dashes=30) if dashed else ln

    def band(self, x0, x1, color=C_HI, opacity=0.12, y0=None, y1=None):
        y0 = self.y_range[0] if y0 is None else y0
        y1 = self.y_range[1] if y1 is None else y1
        a, b = self.p(x0, y0), self.p(x1, y1)
        return Rectangle(width=abs(b[0] - a[0]), height=abs(b[1] - a[1]), fill_color=color, fill_opacity=opacity,
                         stroke_width=0).move_to((a + b) / 2)


def envelope_curve(plot, x, y, color, width=2.0, max_points=1500):
    """Min/max-preserving decimation for dense signals (keeps spikes visible)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) <= max_points:
        return plot.curve(x, y, color, width, max_points)
    n = max_points // 2
    edges = np.linspace(0, len(x), n + 1).astype(int)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = y[a:b]
        i, j = a + int(np.argmin(seg)), a + int(np.argmax(seg))
        for k in sorted((i, j)):
            xs.append(x[k])
            ys.append(y[k])
    return plot.curve(xs, ys, color, width, max_points=10 ** 9)


def bars(values, labels, colors, width=6.0, height=3.0, vmax=None, vmin=0.0, value_fmt=lambda v: f'{v:.2f}',
         label_size=20, value_size=20, gap=0.35, baseline_label=None):
    """Vertical bar chart as a VGroup; ``g.bars[i]`` are the Rectangles (animate with GrowFromEdge(b, DOWN))."""
    n = len(values)
    vmax = vmax if vmax is not None else max(max(values), 1e-9) * 1.15
    bw = (width - gap * (n - 1)) / n
    g, rects, vals, labs = VGroup(), VGroup(), VGroup(), VGroup()
    base_y = 0.0
    for i, (v, l, c) in enumerate(zip(values, labels, colors)):
        h = max(0.02, (v - vmin) / (vmax - vmin) * height)
        x = -width / 2 + bw / 2 + i * (bw + gap)
        r = Rectangle(width=bw, height=h, fill_color=c, fill_opacity=0.9, stroke_width=0)
        r.move_to([x, base_y + h / 2, 0])
        rects.add(r)
        vals.add(T(value_fmt(v).replace('-', '−'), value_size, INK).next_to(r, UP, buff=0.1))
        labs.add(W(l, label_size, DIM, width=chars_for(bw + gap * 0.8, label_size), align=ORIGIN)
                 .next_to([x, base_y, 0], DOWN, buff=0.15))
    axis = Line([-width / 2 - 0.1, base_y, 0], [width / 2 + 0.1, base_y, 0], stroke_color=FAINT, stroke_width=1.6)
    g.add(axis, rects, vals, labs)
    g.bars, g.vals, g.labels, g.axis = rects, vals, labs, axis
    return g


def pie(fracs, colors, radius=1.6, start=np.pi / 2, stroke=BG):
    """Pie of fractions (sum ≈ 1) as Sectors, clockwise from 12 o'clock."""
    g, a = VGroup(), start
    for f, c in zip(fracs, colors):
        ang = -2 * np.pi * f
        s = Sector(radius=radius, start_angle=a, angle=ang, fill_color=c, fill_opacity=0.95, stroke_color=stroke,
                   stroke_width=2)
        g.add(s)
        a += ang
    return g


def load(name):
    """Arrays from data4/<name>.npz, else data/<name>.npz (v1)."""
    p = DATA2 / f'{name}.npz'
    return np.load(p if p.exists() else DATA / f'{name}.npz', allow_pickle=False)


def facts():
    """All facts: v1 (data/facts_*.json) then v2 (data4/facts_*.json), one top-level key per script."""
    out = {}
    for d in (DATA, DATA2):
        for p in sorted(d.glob('facts_*.json')):
            name, data = p.stem[len('facts_'):], json.loads(p.read_text(encoding='utf-8'))
            out[name] = data[name] if list(data) == [name] else data      # facts_y01.json has no wrapper key
    return out


# ── narration-timed beats ────────────────────────────────────────────────────
class BeatScene(Scene):
    """Use ``with self.beat('c01_b01') as dur:`` around each beat's animations. The beat's wav starts at the
    top of the block; the block is padded so the next beat starts GAP seconds after the narration ends.
    ``self.left()`` = narration time still left in the current beat (use it as a final wait/run_time).
    ``self.until(self.cue('some phrase'))`` waits until that phrase of the narration is about to be spoken."""

    def setup(self):
        self.beat_log = []

    def tear_down(self):
        """Write (beat, start, end) in seconds to renders2/beatlog/<Scene>.json, for frame checks."""
        out = VIDEO / 'renders4' / 'beatlog'
        out.mkdir(parents=True, exist_ok=True)
        (out / f'{type(self).__name__}.json').write_text(json.dumps(self.beat_log, indent=1), encoding='utf-8')

    @contextmanager
    def beat(self, bid, gap=GAP):
        dur = BEATS[bid]['seconds']
        t0 = self.renderer.time
        self.add_sound(str(AUDIO / f'{bid}.wav'))
        self._beat = (t0, dur)
        self.beat_log_current = bid
        yield dur
        rest = t0 + dur + gap - self.renderer.time
        if rest > 0.02:
            self.wait(rest)
        self.beat_log.append((bid, t0, self.renderer.time))

    def left(self):
        t0, dur = self._beat
        return max(0.1, t0 + dur - self.renderer.time)

    def cue(self, phrase, bid=None):
        """Fraction (0-1) of the current beat's narration that comes before ``phrase`` (word count), so
        ``self.until(self.cue('For JM'))`` starts an animation roughly when that phrase is spoken."""
        bid = bid or self.beat_log_current
        words = SAY[bid].split()
        target = phrase.split()
        for i in range(len(words)):
            if [w.strip('.,:;!?"').lower() for w in words[i:i + len(target)]] == \
               [w.strip('.,:;!?"').lower() for w in target]:
                return i / max(1, len(words))
        raise ValueError(f'phrase {phrase!r} not in {bid}')

    def until(self, frac):
        """Wait until ``frac`` of the current beat's narration has played (no-op if already past)."""
        t0, dur = self._beat
        rest = t0 + frac * dur - self.renderer.time
        if rest > 0.02:
            self.wait(rest)

    def intro(self, cid, hold=1.3):
        """Chapter opener: big number + title in the centre, then it shrinks into the top-left tag (≈ 2.6 s).
        Call it first thing inside the chapter's first beat; returns the tag (keep it on screen)."""
        n = int(cid[1:])
        big = VGroup(T(f'{n:02d}', 72, C_HI, weight='BOLD'), T(CHAPTER_TITLE[cid], 46, INK)).arrange(DOWN, buff=0.3)
        self.play(FadeIn(big, shift=UP * 0.2), run_time=0.6)
        self.wait(hold)
        tag = chapter_tag(cid)
        self.play(ReplacementTransform(big, tag), run_time=0.7)
        return tag

    def clear_all(self, rt=0.5, keep=()):
        """Fade out everything except the mobjects in ``keep``."""
        gone = [m for m in self.mobjects if m not in keep]
        if gone:
            self.play(*[FadeOut(m) for m in gone], run_time=rt)
